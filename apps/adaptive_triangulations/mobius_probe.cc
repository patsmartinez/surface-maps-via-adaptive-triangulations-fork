/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * MOBIUS PROBE -- how much of the INITIAL residual is a sphere reparameterization?
 *
 * The question. The pipeline embeds each input mesh on the sphere independently, then
 * aligns B's embedding to A's with a single global ROTATION fitted to the landmarks
 * (align_rotation, InitSphereEmbeddings.cc:75 -- 3 DOF). Whatever mismatch that rotation
 * cannot absorb becomes the initial map, and on the L-shape pair the initial residual is
 * ~0.23 while a SHARED embedding gives 2.1e-15. The entire pipeline then moves it by a
 * few percent. So the initialization, not the optimizer, is where the error lives.
 *
 * Mobius transformations are EXACTLY the conformal automorphisms of S^2 (6 DOF), and the
 * embedding energy (surface_to_sphere_angle_energy) is conformal in character. If A and B
 * were isometric, their conformal embeddings would differ by precisely a Mobius
 * transformation. They are not isometric -- that is the point -- but if the
 * non-isometric region is a modest fraction of the area, a Mobius transformation could
 * absorb most of the mismatch. This probe measures whether it does.
 *
 * Four numbers, and the GAPS between them are the result:
 *
 *   R_landmark  3 DOF, fitted to landmarks  -- what the pipeline does today (the baseline)
 *   R_residual  3 DOF, fitted to the residual -- controls for "optimizing the residual
 *               directly" so the 6-DOF gain cannot be confused with a better objective
 *   M_landmark  6 DOF, fitted to landmarks  -- achievable from the information the
 *               pipeline actually has at init
 *   M_residual  6 DOF, fitted to the residual -- the CEILING: the best any Mobius
 *               transformation can do, however one might find it
 *
 * How to read it:
 *   M_residual ~ baseline        -> Mobius is dead. The two embeddings differ by a general
 *                                   diffeomorphism and prescription-driven embedding
 *                                   (priority 2) is unavoidable.
 *   M_residual << baseline, and
 *     M_landmark ~ M_residual    -> cheap fix: replace align_rotation with a Mobius fit.
 *   M_residual << baseline, but
 *     M_landmark ~ baseline      -> the transformation exists; finding it from landmarks
 *                                   does not work, so it needs a residual-driven fit.
 *
 * Nothing here changes the algorithm. The probe only rewrites embeddings_T[B] -- the
 * sphere positions T uses to land on mesh B -- and re-measures. It runs no optimization
 * and no remeshing, so T is still mesh A and the measurement is at full input resolution.
 */
#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/FarthestPointSampling.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedJacobianField.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedResidual.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/MultiRes/MultiResSphereEmbedding.hh>

#include <TinyAD/Utils/Timer.hh>

#include <complex>
#include <fstream>
#include <random>

namespace SurfaceMaps
{

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Overridable from the command line:
///   ./mobius_probe [mesh_dir mesh_A mesh_B [n_fps_landmarks]]
/// Defaults to the L pair at 16 landmarks. Pick a pair baseline_comparison has already
/// run, so the probe's baseline number is literally that run's `00_init` row rather than
/// a re-derived one.
std::string MESH_DIR = "meshes/l_shape";
std::string MESH_A   = "newL.obj";
std::string MESH_B   = "stretchedL.obj";
int N_FPS_LANDMARKS  = 16;

/// Reuse the embeddings baseline_comparison cached for its AonB_prescribed config, so
/// the baseline residual matches that CSV exactly instead of depending on a fresh
/// multi_res_sphere_embedding (which is FP-sensitive). Copied into the probe's own
/// output dir before init_map runs, so nothing is written back into the baseline dir.
/// Set to false to compute fresh embeddings here instead.
const bool REUSE_BASELINE_EMBEDDINGS = true;

/// Nelder-Mead budget per start. The residual objective costs one pass over T's faces
/// (6 lifts + a BSP lookup + a 2x2 SVD each), which is the expensive one.
const int  NM_MAX_EVALS_RESIDUAL = 1200;
const int  NM_MAX_EVALS_LANDMARK = 20000; // landmark objective is ~16 points, so cheap
const int  N_RANDOM_STARTS       = 4;     // besides the identity start
const bool WRITE_FITTED_EMBEDDING = true;

// ---------------------------------------------------------------------------
// Mobius transformations of S^2
// ---------------------------------------------------------------------------

using Cplx = std::complex<double>;

/// A Mobius transformation of the sphere, as an element of SL(2,C) acting through
/// stereographic projection. Verified numerically before being written here: identity to
/// 4e-16, image norm 1 to 4e-16 over 200 random SL(2,C) x 504 points, SU(2) reproduces
/// the corresponding real rotation to 5e-16, both exact poles correct, conformal, and the
/// group law holds to 1e-15.
struct Mobius
{
    Cplx a{1.0, 0.0}, b{0.0, 0.0}, c{0.0, 0.0}, d{1.0, 0.0};

    /// Scale to det = 1. The action only depends on the matrix up to a complex factor, so
    /// this changes nothing geometrically; it just keeps the entries well-conditioned
    /// while the search wanders.
    bool normalize()
    {
        const Cplx det = a * d - b * c;
        if (!(std::abs(det) > 1e-12))
            return false; // degenerate: the whole sphere collapses to a point
        const Cplx s = std::sqrt(det);
        a /= s; b /= s; c /= s; d /= s;
        return true;
    }

    /// Apply to a point of S^2.
    ///
    /// Stereographic projection is carried in HOMOGENEOUS form, (p : q) with w = p/q, so
    /// there is no division anywhere except by |p|^2 + |q|^2, and no special case for a
    /// point mapping to infinity. Both (x + iy, 1 - z) and (1 + z, x - iy) represent
    /// sigma(point) -- they agree because (x + iy)(x - iy) = 1 - z^2 = (1 + z)(1 - z) --
    /// and each is degenerate at exactly one pole, so pick by the sign of z and neither
    /// pole is ever a special case.
    Vec3d operator() (const Vec3d& _p) const
    {
        Cplx p, q;
        if (_p[2] <= 0.0) { p = Cplx(_p[0], _p[1]);  q = Cplx(1.0 - _p[2], 0.0); } // |1-z| >= 1
        else              { p = Cplx(1.0 + _p[2], 0.0); q = Cplx(_p[0], -_p[1]); } // |1+z| >= 1

        const Cplx p2 = a * p + b * q;
        const Cplx q2 = c * p + d * q;

        const double np = std::norm(p2); // |p2|^2
        const double nq = std::norm(q2);
        const double n  = np + nq;
        if (!(n > 0.0))
            return _p; // only if the matrix is singular; normalize() already rejects that

        const Cplx pq = p2 * std::conj(q2);
        return Vec3d(2.0 * pq.real() / n, 2.0 * pq.imag() / n, (np - nq) / n);
    }
};

/// 6-DOF chart on PSL(2,C), centred on the identity: d is pinned to 1 and (a, b, c) are
/// free complex numbers. Covers everything with d != 0, which is all we need -- the
/// search starts at the identity (1, 0, 0, 0, 0, 0) and the excluded locus is far away.
Mobius mobius_from_params(const Eigen::Matrix<double, 6, 1>& _x)
{
    Mobius m;
    m.a = Cplx(_x[0], _x[1]);
    m.b = Cplx(_x[2], _x[3]);
    m.c = Cplx(_x[4], _x[5]);
    m.d = Cplx(1.0, 0.0);
    return m;
}

Eigen::Matrix<double, 6, 1> mobius_identity_params()
{
    Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Zero();
    x[0] = 1.0; // a = 1, b = c = 0, d = 1
    return x;
}

/// 3-DOF rotation, as a Mobius transformation so both fits go through one code path.
/// Rodrigues on the axis-angle vector _v, then the real 3x3 matrix is applied directly --
/// no need to detour through SU(2).
struct Rotation
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Vec3d operator() (const Vec3d& _p) const { return R * _p; }
};

Rotation rotation_from_params(const Eigen::Vector3d& _v)
{
    Rotation r;
    const double theta = _v.norm();
    if (theta < 1e-14)
        return r;
    r.R = Eigen::AngleAxisd(theta, _v / theta).toRotationMatrix();
    return r;
}

// ---------------------------------------------------------------------------
// Nelder-Mead
// ---------------------------------------------------------------------------

/// Derivative-free on purpose. The residual objective is non-smooth twice over: it sums
/// |s - 1|, and the prescription lookup picks a mesh-A face by a BSP query, so the
/// objective is piecewise with jumps wherever a T centroid crosses an input face
/// boundary. Gradients would be fiction.
template <int N, typename F>
double nelder_mead(
        F&& _f,
        Eigen::Matrix<double, N, 1>& _x,
        const double _initial_step,
        const int _max_evals)
{
    using Vec = Eigen::Matrix<double, N, 1>;

    std::vector<Vec> simplex(N + 1, _x);
    for (int i = 0; i < N; ++i)
        simplex[i + 1][i] += _initial_step;

    std::vector<double> f(N + 1);
    int evals = 0;
    for (int i = 0; i <= N; ++i) { f[i] = _f(simplex[i]); ++evals; }

    auto order = [&] ()
    {
        std::vector<int> idx(N + 1);
        for (int i = 0; i <= N; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&] (int p, int q) { return f[p] < f[q]; });
        std::vector<Vec> s2(N + 1);
        std::vector<double> f2(N + 1);
        for (int i = 0; i <= N; ++i) { s2[i] = simplex[idx[i]]; f2[i] = f[idx[i]]; }
        simplex = s2; f = f2;
    };
    order();

    while (evals < _max_evals)
    {
        // Converged when the simplex has collapsed.
        double spread = 0.0;
        for (int i = 1; i <= N; ++i)
            spread = std::max(spread, (simplex[i] - simplex[0]).norm());
        if (spread < 1e-10)
            break;

        Vec centroid = Vec::Zero();
        for (int i = 0; i < N; ++i) centroid += simplex[i];
        centroid /= (double)N;

        const Vec worst = simplex[N];
        const Vec reflected = centroid + 1.0 * (centroid - worst);
        const double f_r = _f(reflected); ++evals;

        if (f_r < f[0])
        {
            const Vec expanded = centroid + 2.0 * (centroid - worst);
            const double f_e = _f(expanded); ++evals;
            if (f_e < f_r) { simplex[N] = expanded; f[N] = f_e; }
            else           { simplex[N] = reflected; f[N] = f_r; }
        }
        else if (f_r < f[N - 1])
        {
            simplex[N] = reflected; f[N] = f_r;
        }
        else
        {
            const Vec contracted = centroid + 0.5 * ((f_r < f[N] ? reflected : worst) - centroid);
            const double f_c = _f(contracted); ++evals;
            if (f_c < std::min(f_r, f[N])) { simplex[N] = contracted; f[N] = f_c; }
            else
            {
                for (int i = 1; i <= N; ++i)
                {
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0]);
                    f[i] = _f(simplex[i]); ++evals;
                }
            }
        }
        order();
    }

    _x = simplex[0];
    return f[0];
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

/// Push the whole of mesh B's T-embedding through _map and re-measure the residual.
///
/// PENALTY: the residual skips T faces that are flipped on either side, so a search that
/// minimizes the mean could cheat by flipping the worst faces out of the statistic. Any
/// drop below the baseline face count is charged at 10 per unit of lost fraction, which
/// dwarfs the residual's own range and makes flipping strictly unprofitable.
template <typename Map>
double residual_objective(
        MapState& _map_state,
        const ExternalProperty<VH, Vec3d>& _embedding_T_B_initial,
        const int _mesh_B_idx,
        const int _baseline_n_measured,
        const Map& _map,
        ResidualStats* _out = nullptr)
{
    for (auto vh : _map_state.mesh_T.vertices())
        _map_state.embeddings_T[_mesh_B_idx][vh] = _map(_embedding_T_B_initial[vh]).normalized();

    ResidualStats res;
    compute_metric_residual(_map_state, res);
    if (_out) *_out = res;

    if (res.n_faces == 0)
        return 1e9;

    const int total = res.n_faces + res.n_faces_skipped;
    const double lost = (total > 0)
            ? std::max(0.0, (double)(_baseline_n_measured - res.n_faces) / (double)total)
            : 1.0;

    return res.mean_aw + 10.0 * lost;
}

/// Sum of squared chordal distances between each landmark of A and its partner on B,
/// both on the sphere, after _map is applied to A's sphere positions.
///
/// Direction: at init a point of A travels emb_A -> (T's sphere coords) -> emb_B^-1, so
/// landmark i of A lands on lm_B[i] exactly when _map(emb_A[lm_A[i]]) == emb_B[lm_B[i]].
/// With _map restricted to rotations this is the objective align_rotation already solves,
/// so the 3-DOF landmark fit reproduces the pipeline and is the honest baseline.
template <typename Map>
double landmark_objective(
        const MapState& _map_state,
        const Map& _map)
{
    const auto& lm_A = _map_state.landmarks_input[0];
    const auto& lm_B = _map_state.landmarks_input[1];

    double sum = 0.0;
    for (int i = 0; i < (int)lm_A.size(); ++i)
    {
        const Vec3d target = _map_state.embeddings_input[1][lm_B[i]];
        const Vec3d mapped = _map(_map_state.embeddings_input[0][lm_A[i]]).normalized();
        sum += (mapped - target).squaredNorm();
    }
    return sum / (double)lm_A.size();
}

// ---------------------------------------------------------------------------

struct FitResult
{
    std::string name;
    int dof = 0;
    std::string fitted_to;
    double landmark_rms = 0.0; ///< sqrt of the landmark objective, in sphere chord length
    ResidualStats res;
};

void print_table(const std::vector<FitResult>& _rows)
{
    ISM_INFO("");
    ISM_INFO("=================================== MOBIUS PROBE ===================================");
    ISM_INFO(MESH_DIR << "  " << MESH_A << " -> " << MESH_B << "   fps" << N_FPS_LANDMARKS);
    ISM_INFO("");
    ISM_INFO("fit          dof  fitted to    resid_mean_aw  median_aw  frac>0.1_aw   area_aw  shape_aw   lm_rms  faces");
    for (const auto& r : _rows)
    {
        std::ostringstream os;
        os << std::left << std::setw(13) << r.name
           << std::setw(5) << r.dof
           << std::setw(13) << r.fitted_to
           << std::right << std::fixed << std::setprecision(6)
           << std::setw(13) << r.res.mean_aw
           << std::setw(11) << r.res.median_aw
           << std::setw(13) << r.res.frac_gt_0p1_aw
           << std::setw(10) << r.res.area_part_aw
           << std::setw(10) << r.res.shape_part_aw
           << std::setw(9) << r.landmark_rms
           << std::setw(7) << r.res.n_faces;
        ISM_INFO(os.str());
    }
    ISM_INFO("====================================================================================");

    if (_rows.size() >= 4)
    {
        const double base = _rows[0].res.mean_aw;
        ISM_INFO("");
        ISM_INFO("Reduction vs the pipeline's own initialization (R_landmark = " << base << "):");
        for (size_t i = 1; i < _rows.size(); ++i)
            ISM_INFO("  " << _rows[i].name << ": " << base / std::max(1e-30, _rows[i].res.mean_aw)
                     << "x  (to " << _rows[i].res.mean_aw << ")");
    }
}

void run()
{
    const fs::path output_root =
            OUTPUT_PATH / ("mobius_probe_" + fs::path(MESH_DIR).filename().string());
    fs::create_directories(output_root);

    const fs::path mesh_path_A = DATA_PATH / MESH_DIR / MESH_A;
    const fs::path mesh_path_B = DATA_PATH / MESH_DIR / MESH_B;

    // --- Landmarks: same FPS selection baseline_comparison uses -----------
    const fs::path landmarks_path_A = output_root / "landmarks_A.pinned";
    const fs::path landmarks_path_B = output_root / "landmarks_B.pinned";
    {
        const TriMesh mesh_A = read_mesh(mesh_path_A);
        const std::vector<VH> landmarks = farthest_point_sampling(mesh_A, N_FPS_LANDMARKS);
        write_landmarks_pinned(landmarks, landmarks_path_A, {}, true);
        write_landmarks_pinned(landmarks, landmarks_path_B, {}, true);
    }

    // --- Embeddings -------------------------------------------------------
    const fs::path embedding_path_A = output_root / "embedding_A.obj";
    const fs::path embedding_path_B = output_root / "embedding_B.obj";
    if (REUSE_BASELINE_EMBEDDINGS)
    {
        const fs::path src = OUTPUT_PATH
                / ("baseline_comparison_" + fs::path(MESH_DIR).filename().string())
                / ("AonB_prescribed_fps" + std::to_string(N_FPS_LANDMARKS));
        ISM_ASSERT(fs::exists(src / "embedding_A.obj"));
        ISM_ASSERT(fs::exists(src / "embedding_B.obj"));
        fs::copy_file(src / "embedding_A.obj", embedding_path_A, fs::copy_options::overwrite_existing);
        fs::copy_file(src / "embedding_B.obj", embedding_path_B, fs::copy_options::overwrite_existing);
        ISM_INFO("Reusing embeddings from " << src);
    }

    // --- Init: identical to baseline_comparison up to its "00_init" report
    MapState map_state;
    init_map(map_state,
             { mesh_path_A, mesh_path_B },
             { landmarks_path_A, landmarks_path_B },
             { embedding_path_A, embedding_path_B },
             false);
    map_state.set_distortion_pairs(DistortionPairs::All);
    map_state.prescribed_jacobians.push_back(
                extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]));
    assign_vertices_to_T_faces(map_state);

    const int mesh_B_idx = map_state.pairs_map_distortion[0].second;

    // init_map sets embeddings_T[i] = embeddings_input[0] for EVERY i, so both of T's
    // sphere position sets start as mesh A's. This is the one the probe rewrites.
    const ExternalProperty<VH, Vec3d> embedding_T_B_initial = map_state.embeddings_T[mesh_B_idx];

    ISM_INFO("T has " << map_state.mesh_T.n_vertices() << " vertices, "
             << map_state.mesh_T.n_faces() << " faces (= mesh A; no optimization has run)");

    // --- Baseline: the pipeline's own state, R = identity on top of the
    //     rotation align_rotation already applied -------------------------
    std::vector<FitResult> rows;
    int baseline_n_measured = 0;
    {
        FitResult r;
        r.name = "R_landmark"; r.dof = 3; r.fitted_to = "landmarks";
        const Rotation id;
        residual_objective(map_state, embedding_T_B_initial, mesh_B_idx, 0, id, &r.res);
        r.landmark_rms = std::sqrt(landmark_objective(map_state, id));
        baseline_n_measured = r.res.n_faces;
        rows.push_back(r);
        ISM_INFO("Baseline (pipeline init): residual_mean_aw = " << r.res.mean_aw
                 << ", measured " << r.res.n_faces << " / "
                 << (r.res.n_faces + r.res.n_faces_skipped) << " faces");
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.35, 0.35);

    // --- R_residual: best ROTATION for the residual (3 DOF) ---------------
    // The control. If the 6-DOF fit beats the baseline only as much as this does, the
    // gain was "we optimized the residual directly", not "Mobius has more freedom".
    {
        TinyAD::Timer t("R_residual");
        auto obj = [&] (const Eigen::Vector3d& _v)
        {
            return residual_objective(map_state, embedding_T_B_initial, mesh_B_idx,
                                      baseline_n_measured, rotation_from_params(_v));
        };
        Eigen::Vector3d best = Eigen::Vector3d::Zero();
        double best_f = obj(best);
        for (int s = 0; s <= N_RANDOM_STARTS; ++s)
        {
            Eigen::Vector3d x = Eigen::Vector3d::Zero();
            if (s > 0) for (int i = 0; i < 3; ++i) x[i] = jitter(rng);
            const double f = nelder_mead<3>(obj, x, 0.25, NM_MAX_EVALS_RESIDUAL);
            if (f < best_f) { best_f = f; best = x; }
            ISM_INFO("  R_residual start " << s << ": " << f);
        }
        FitResult r;
        r.name = "R_residual"; r.dof = 3; r.fitted_to = "residual";
        const Rotation rot = rotation_from_params(best);
        residual_objective(map_state, embedding_T_B_initial, mesh_B_idx, baseline_n_measured, rot, &r.res);
        r.landmark_rms = std::sqrt(landmark_objective(map_state, rot));
        rows.push_back(r);
        t.stop();
    }

    // --- M_landmark: 6-DOF Mobius fitted to the landmarks -----------------
    // The drop-in replacement for align_rotation: same information, more freedom.
    {
        TinyAD::Timer t("M_landmark");
        auto obj = [&] (const Eigen::Matrix<double, 6, 1>& _p)
        {
            Mobius m = mobius_from_params(_p);
            if (!m.normalize()) return 1e9;
            return landmark_objective(map_state, m);
        };
        Eigen::Matrix<double, 6, 1> best = mobius_identity_params();
        double best_f = obj(best);
        for (int s = 0; s <= N_RANDOM_STARTS; ++s)
        {
            Eigen::Matrix<double, 6, 1> x = mobius_identity_params();
            if (s > 0) for (int i = 0; i < 6; ++i) x[i] += jitter(rng);
            const double f = nelder_mead<6>(obj, x, 0.2, NM_MAX_EVALS_LANDMARK);
            if (f < best_f) { best_f = f; best = x; }
        }
        FitResult r;
        r.name = "M_landmark"; r.dof = 6; r.fitted_to = "landmarks";
        Mobius m = mobius_from_params(best);
        ISM_ASSERT(m.normalize());
        residual_objective(map_state, embedding_T_B_initial, mesh_B_idx, baseline_n_measured, m, &r.res);
        r.landmark_rms = std::sqrt(landmark_objective(map_state, m));
        rows.push_back(r);
        ISM_INFO("  M_landmark landmark objective: " << best_f << " (rms " << r.landmark_rms << ")");
        t.stop();
    }

    // --- M_residual: 6-DOF Mobius fitted to the residual ------------------
    // The ceiling. Starts from the identity, from random jitters, AND from the
    // landmark-fitted transformation, because that is the best informed guess available.
    Mobius best_mobius;
    {
        TinyAD::Timer t("M_residual");
        auto obj = [&] (const Eigen::Matrix<double, 6, 1>& _p)
        {
            Mobius m = mobius_from_params(_p);
            if (!m.normalize()) return 1e9;
            return residual_objective(map_state, embedding_T_B_initial, mesh_B_idx,
                                      baseline_n_measured, m);
        };

        // Seed list: identity, the landmark fit, then random jitters.
        std::vector<Eigen::Matrix<double, 6, 1>> seeds;
        seeds.push_back(mobius_identity_params());
        {
            auto lm_obj = [&] (const Eigen::Matrix<double, 6, 1>& _p)
            {
                Mobius m = mobius_from_params(_p);
                if (!m.normalize()) return 1e9;
                return landmark_objective(map_state, m);
            };
            Eigen::Matrix<double, 6, 1> x = mobius_identity_params();
            nelder_mead<6>(lm_obj, x, 0.2, NM_MAX_EVALS_LANDMARK);
            seeds.push_back(x);
        }
        for (int s = 0; s < N_RANDOM_STARTS; ++s)
        {
            Eigen::Matrix<double, 6, 1> x = mobius_identity_params();
            for (int i = 0; i < 6; ++i) x[i] += jitter(rng);
            seeds.push_back(x);
        }

        Eigen::Matrix<double, 6, 1> best = seeds[0];
        double best_f = obj(best);
        for (int s = 0; s < (int)seeds.size(); ++s)
        {
            Eigen::Matrix<double, 6, 1> x = seeds[s];
            const double f = nelder_mead<6>(obj, x, 0.2, NM_MAX_EVALS_RESIDUAL);
            ISM_INFO("  M_residual seed " << s << ": " << f);
            if (f < best_f) { best_f = f; best = x; }
        }

        FitResult r;
        r.name = "M_residual"; r.dof = 6; r.fitted_to = "residual";
        best_mobius = mobius_from_params(best);
        ISM_ASSERT(best_mobius.normalize());
        residual_objective(map_state, embedding_T_B_initial, mesh_B_idx, baseline_n_measured,
                           best_mobius, &r.res);
        r.landmark_rms = std::sqrt(landmark_objective(map_state, best_mobius));
        rows.push_back(r);
        t.stop();
    }

    // --- Reference: a SHARED embedding, the unreachable lower bound --------
    // Both meshes share connectivity, so giving B mesh A's embedding makes the initial
    // map exactly the ground-truth correspondence and the residual ~0. Not a fit -- it is
    // the number the fits above are trying to approach, printed so the scale is visible.
    {
        FitResult r;
        r.name = "(shared emb)"; r.dof = 0; r.fitted_to = "ground truth";
        MapState gt = map_state;
        // MapState's copy constructor does not copy pairs_map_distortion (nor
        // active_for_approx) -- see MapState.hh. Restore it or compute_metric_residual
        // has no pair to measure.
        gt.set_distortion_pairs(DistortionPairs::All);
        gt.embeddings_input[1] = gt.embeddings_input[0];
        gt.meshes_embeddings_input[1] = embedding_to_mesh(gt.meshes_input[1], gt.embeddings_input[1]);
        gt.compute_bsp_trees();
        for (auto vh : gt.mesh_T.vertices())
            gt.embeddings_T[mesh_B_idx][vh] = embedding_T_B_initial[vh];
        compute_metric_residual(gt, r.res);
        rows.push_back(r);
    }

    print_table(rows);

    // --- Outputs ----------------------------------------------------------
    std::ofstream csv(output_root / ("mobius_probe_fps" + std::to_string(N_FPS_LANDMARKS) + ".csv"));
    ISM_INFO("Writing " << (output_root / ("mobius_probe_fps" + std::to_string(N_FPS_LANDMARKS) + ".csv")));
    csv << "fit,dof,fitted_to,residual_mean_aw,residual_median_aw,residual_frac_gt_0p1_aw,"
           "residual_area_aw,residual_shape_aw,residual_max,landmark_rms,n_faces,n_faces_skipped\n";
    for (const auto& r : rows)
        csv << r.name << "," << r.dof << "," << r.fitted_to << ","
            << r.res.mean_aw << "," << r.res.median_aw << "," << r.res.frac_gt_0p1_aw << ","
            << r.res.area_part_aw << "," << r.res.shape_part_aw << "," << r.res.max << ","
            << r.landmark_rms << "," << r.res.n_faces << "," << r.res.n_faces_skipped << "\n";
    csv.close();

    if (WRITE_FITTED_EMBEDDING)
    {
        // B's embedding pushed through the best Mobius transformation, so the fitted
        // initialization can be fed straight back into a run. The probe transforms T's
        // B-side positions; the equivalent change to the INPUT embedding is the inverse,
        // because the lift goes emb_B^-1(mu(.)): mu(emb_A) == emb_B means
        // emb_B_new = mu^-1(emb_B) leaves emb_B_new^-1(emb_A) at the fitted map.
        Mobius inv;
        inv.a =  best_mobius.d; inv.b = -best_mobius.b;
        inv.c = -best_mobius.c; inv.d =  best_mobius.a;
        ISM_ASSERT(inv.normalize());

        ExternalProperty<VH, Vec3d> fitted(map_state.meshes_input[1]);
        for (auto vh : map_state.meshes_input[1].vertices())
            fitted[vh] = inv(map_state.embeddings_input[1][vh]).normalized();
        write_embedding(map_state.meshes_input[1], fitted, output_root / "embedding_B_mobius_fitted.obj");
        ISM_INFO("Wrote embedding_B_mobius_fitted.obj (B's embedding pulled back through the fit)");
    }

    ISM_INFO("Done.");
}

}  // namespace SurfaceMaps

int main(int argc, char** argv)
{
    using namespace SurfaceMaps;

    if (argc >= 4)
    {
        MESH_DIR = argv[1];
        MESH_A   = argv[2];
        MESH_B   = argv[3];
        if (argc >= 5)
            N_FPS_LANDMARKS = std::atoi(argv[4]);
    }
    else if (argc != 1)
    {
        ISM_ERROR_throw("usage: mobius_probe [mesh_dir mesh_A mesh_B [n_fps_landmarks]]");
    }

    init_lib_surface_maps();
    run();
    return 0;
}
