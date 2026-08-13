/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Baseline comparison for the prescribed Jacobian work.
 *
 * Runs 4 configurations over the L-shape meshes:
 *
 *                          | Symmetric Dirichlet | Prescribed metric
 *   -----------------------+---------------------+-------------------
 *   A on A (newL -> newL)  |  consistency check  |  consistency check
 *   A on B (newL -> strL)  |  baseline           |  the real run
 *
 * "A on A" maps a mesh to ITSELF. The correct answer is the identity map, and the
 * optimization must not deviate from it -- a pure consistency check.
 *
 * It is also a sharp test of the new energy: when both meshes are equal we have
 * J* = I everywhere, so the prescribed metric energy reduces EXACTLY to symmetric
 * Dirichlet (tr(M) + tr(M^-1) = ||J||^2 + ||J^-1||^2). The two energy columns must
 * therefore agree on the A-on-A row; if they disagree, the prescribed path has a bug.
 * (For J = I the SVD is degenerate and U, V are arbitrary, but M* = V*I*V^T = I
 * regardless -- the metric form is immune to that ambiguity.)
 *
 * Ground-truth correspondence error is the primary metric: the metric residual is
 * what we minimize (a training loss), and under the gauge freedom of a metric
 * prescription it can be ~0 while the map is visibly wrong.
 *
 * Toggle FPS landmarks with the flags below; re-run to get another 4 results.
 */
#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/FarthestPointSampling.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/Viewer/MeshView.hh>
#include <SurfaceMaps/Viewer/Colors.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedJacobianField.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeWithRemeshing.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Visualization.hh>
#include <SurfaceMaps/MultiRes/MultiResSphereEmbedding.hh>

#include <Eigen/SVD>
#include <TinyAD/Utils/Timer.hh>
#include <fstream>

namespace SurfaceMaps
{

// ---------------------------------------------------------------------------
// Settings. Toggle these and re-run to get another set of 4 results.
// ---------------------------------------------------------------------------

const bool USE_FPS_LANDMARKS = false; // false = default landmarks (first 3 vertices)
const int  N_FPS_LANDMARKS   = 8;     // only used when USE_FPS_LANDMARKS is true
const bool OPEN_VIEWER       = true;  // false = write screenshots instead

/// Optional ablation, applies to the A-on-B configs only: give mesh B the SAME sphere
/// embedding as mesh A. The two L meshes share connectivity, so the initial map is then
/// exactly the ground-truth correspondence, and the question becomes whether the energy
/// HOLDS the map there. Isolates energy correctness from initialization quality.
const bool START_FROM_GROUND_TRUTH = false;

/// Run the COARSE phase with the prescribed energy too, not just the final phase.
/// By default the coarse phase uses standard symmetric Dirichlet in every config,
/// whose minimum for A->B is the as-isometric-as-possible map -- i.e. it actively
/// pulls away from the prescription, and the final phase then has to undo that.
const bool PRESCRIBED_IN_COARSE_PHASE = false;

/// Disable all remeshing in the coarse AND final phases (T keeps the connectivity it
/// has after the landmark phase). Diagnostic for whether remeshing is what breaks the
/// map: if the A-on-A correspondence error stays at ~0 with this on, remeshing is the
/// culprit; if it still drifts, the continuous optimization is.
/// NOTE: with this on, T stays tiny (~15 vertices), so the metric residual is dominated
/// by coarse-scale representation error -- read corr_mean, not residual.
const bool DISABLE_REMESHING = false;

/// Weight of the map-distortion term -- the ONLY term the prescription lives in.
/// The total objective is
///     E_barrier*w_barrier + E_approx*w_approx + E_map*w_map + E_mesh*w_mesh
/// and both coarse_phase_settings() and fine_phase_settings() ship
/// w_map = w_mesh = w_approx = 1.0, so the prescription is one of three unit-weighted
/// votes -- and the other two reward exactly what we observe improving (surface
/// approximation, mesh quality) while being indifferent to where area goes.
/// Sweep 1 / 10 / 100: residual drops => the prescription was outvoted, and the fix is
/// this line; residual flat => it is being heard and ignored, i.e. the basin problem.
/// Applied to BOTH energy modes on purpose, so the comparison controls for "more map
/// weight helps regardless of WHICH map energy".
/// Raise this rather than lowering w_approx: w_approx > 0 also gates
/// update_assignment_vertices_to_T_faces inside the line search, so zeroing it would
/// change the algorithm structurally instead of just reweighting it.
const double W_MAP = 1.0;

/// Per-iteration trace. The optimizer already exposes two callbacks and we just pass
/// lambdas into them -- nothing in the algorithm changes.
///   _callback_for_optim  fires once per Newton iteration, inside the loop, after
///                        embeddings_T and the local bases have been updated
///                        (OptimizeMap.cc: "after_optimization").
///   _callback_for_remesh fires three times per remeshing pass
///                        (Remeshing.cc: "after_splits" / "after_collapses" / "after_flips").
/// Because they are separate, a jump in the metrics can be attributed to a continuous
/// Newton step or to one specific remeshing operation -- which no end-of-phase number
/// can tell you. Writes <config>/trace.csv, one row per step.
const bool TRACE_ITERATIONS = true;
/// Also dump lifted meshes per traced step, for an animation. Off-by-default would make
/// the trace CSV-only; the meshes are what you actually watch.
const bool TRACE_WRITE_MESHES = true;
/// Dump a mesh only every Nth traced step (the CSV always gets every step).
/// Raise this if the frame count gets unwieldy; the coarse phase is the interesting part.
const int TRACE_MESH_STRIDE = 1;

/// Checkerboard transfer settings.
/// Texcoords are built as (point[(dir+1)%3], point[(dir+2)%3]), so dir = 2 gives
/// (x, y) -- the right choice for the L, which lies in the xy-plane and is extruded
/// along z. (dir = 1 would give (z, x), i.e. constant across the L's main faces,
/// which degenerates the checkerboard into stripes.)
/// Raise TEXTURE_FACTOR if the checkers are too coarse: init_map normalizes the
/// meshes to unit surface area, so their coordinates only span ~0.5.
const int    TEXTURE_PROJECTION_DIR = 2;
const double TEXTURE_FACTOR         = 4.0;

// ---------------------------------------------------------------------------

enum class PairMode
{
    SelfMap,   // "A on A": newL -> newL. Correct answer is the identity map.
    Stretched, // "A on B": newL -> stretchedL. The actual problem.
};

enum class EnergyMode
{
    SymmetricDirichlet, // Original algorithm
    PrescribedMetric,   // Metric-form prescribed Jacobian
};

struct Config
{
    PairMode pair_mode;
    EnergyMode energy_mode;

    std::string name() const
    {
        const std::string pair = (pair_mode == PairMode::SelfMap) ? "AonA" : "AonB";
        const std::string energy = (energy_mode == EnergyMode::SymmetricDirichlet) ? "dirichlet" : "prescribed";
        const std::string lm = USE_FPS_LANDMARKS ? ("fps" + std::to_string(N_FPS_LANDMARKS)) : "default3";
        // The sphere embedding is cached per output dir, and START_FROM_GROUND_TRUTH
        // changes WHICH embedding is used -- so it must be part of the name, or a
        // stale cache from a previous run is silently reused and the ablation is a no-op.
        const std::string gt = (pair_mode == PairMode::Stretched && START_FROM_GROUND_TRUTH) ? "_gt" : "";
        // Same reasoning as gt: w_map changes the result, so it must change the output
        // dir, or a cached embedding from a different weight gets reused silently.
        const std::string wm = (W_MAP == 1.0) ? "" : ("_wmap" + std::to_string((int)W_MAP));
        return pair + "_" + energy + "_" + lm + gt + wm;
    }
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Ground-truth correspondence error.
/// The two L meshes share connectivity, so the true image of vertex v of mesh A is
/// vertex v of mesh B. (In the A-on-A self-map case the two meshes are identical, so
/// this is the identity map and the error must stay at ~0.)
/// Errors are normalized by sqrt(area of B) so they are
/// comparable across meshes (standard in the correspondence literature).
/// This is the primary metric: unlike the metric residual, it is not the quantity
/// being optimized, and it detects material "sliding" that leaves the residual at ~0.
void compute_correspondence_error(
        const MapState& _map_state,
        double& _mean_error,
        double& _median_error,
        double& _max_error)
{
    const TriMesh mapped_A = map_vertices_to_target(_map_state, 0, 1);
    const TriMesh& mesh_B = _map_state.meshes_input[1];
    ISM_ASSERT_EQ(mapped_A.n_vertices(), mesh_B.n_vertices());

    // Normalization: sqrt of B's surface area
    double area_B = 0.0;
    for (auto fh : mesh_B.faces())
        area_B += mesh_B.calc_face_area(fh);
    const double scale = 1.0 / std::sqrt(area_B);

    std::vector<double> errors;
    errors.reserve(mesh_B.n_vertices());
    for (auto vh : mesh_B.vertices())
        errors.push_back((mapped_A.point(vh) - mesh_B.point(vh)).norm() * scale);

    _mean_error = 0.0;
    _max_error = 0.0;
    for (const double e : errors)
    {
        _mean_error += e;
        _max_error = std::max(_max_error, e);
    }
    _mean_error /= (double)errors.size();

    std::sort(errors.begin(), errors.end());
    _median_error = errors[errors.size() / 2];
}

/// Distribution of the per-face residual.
///
/// Two things the plain mean cannot do:
///
/// (1) COMPARE ACROSS STAGES. The unweighted average over T faces changes meaning when
///     refinement takes T from ~50 faces to ~630, because refinement concentrates small
///     faces wherever it chooses to refine. The *_aw fields weight each face by the area
///     of the mesh-A material it covers, which is stage-independent; compare those.
///     The unweighted fields are kept unchanged so older CSVs remain comparable, and the
///     gap between `mean` and `mean_aw` is itself a readout of how much the unweighted
///     statistic is being distorted.
///
/// (2) SAY WHAT KIND OF WRONG. The residual J*J*^-1 has singular values s0, s1, and
///     log-splits into two independent parts:
///         area  = |log(s0*s1)|  -- material is in the wrong PLACE (mis-allocated area)
///         shape = |log(s0/s1)|  -- material is the wrong SHAPE   (sheared / anisotropy off)
///     Both vanish exactly at s0 = s1 = 1. Area-dominant points at coarse-scale
///     representativeness (the centroid lookup smearing the prescription); shape-dominant
///     points at local shear, i.e. sliding and initialization.
struct ResidualStats
{
    double mean = 0.0;            ///< unweighted; unchanged, comparable with earlier runs
    double mean_aw = 0.0;         ///< area-weighted -- the one to compare ACROSS stages
    double median = 0.0;
    double median_aw = 0.0;
    double p90 = 0.0;
    double max = 0.0;             ///< per-face max over the two singular values, then max over faces
    double frac_gt_0p1 = 0.0;     ///< fraction of faces off by more than 10%
    double frac_gt_0p1_aw = 0.0;  ///< fraction of AREA off by more than 10%
    double area_part_aw = 0.0;    ///< area-weighted mean |log(s0*s1)|
    double shape_part_aw = 0.0;   ///< area-weighted mean |log(s0/s1)|
    int n_faces = 0;              ///< faces actually measured; degenerate ones are skipped
};

/// Percentile of (value, weight) pairs, sorted ascending by value.
double weighted_percentile(
        const std::vector<std::pair<double, double>>& _sorted,
        const double _total_weight,
        const double _q)
{
    double acc = 0.0;
    for (const auto& vw : _sorted)
    {
        acc += vw.second;
        if (acc >= _q * _total_weight)
            return vw.first;
    }
    return _sorted.back().first;
}

/// Rotation-invariant residual against the prescribed field: singular values of
/// J * J*^{-1}, which are all 1 exactly when the pullback metric matches the
/// prescription. Computed for ALL configs (the baselines need it for evaluation
/// even though they do not optimize it).
void compute_metric_residual(
        const MapState& _map_state,
        ResidualStats& _stats)
{
    _stats = ResidualStats();

    // (deviation, area of the lifted A triangle), kept so we can take percentiles
    // and area-weighted means at the end.
    std::vector<std::pair<double, double>> devs;
    devs.reserve(_map_state.mesh_T.n_faces());
    double total_area = 0.0;

    const int mesh_A_idx = _map_state.pairs_map_distortion[0].first;
    const int mesh_B_idx = _map_state.pairs_map_distortion[0].second;

    for (auto fh : _map_state.mesh_T.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_map_state.mesh_T, fh, vh_a, vh_b, vh_c);

        const Vec3d a_sphere = _map_state.embeddings_T[0][vh_a];
        const Vec3d b_sphere = _map_state.embeddings_T[0][vh_b];
        const Vec3d c_sphere = _map_state.embeddings_T[0][vh_c];

        auto lift = [&] (const Vec3d& _p, const int _idx)
        {
            return lift_vertex_to_surface(_p, _map_state.meshes_input[_idx],
                _map_state.meshes_embeddings_input[_idx], _map_state.bsp_embeddings_input[_idx]);
        };

        const Vec3d a_lifted_A = lift(a_sphere, mesh_A_idx);
        const Vec3d b_lifted_A = lift(b_sphere, mesh_A_idx);
        const Vec3d c_lifted_A = lift(c_sphere, mesh_A_idx);
        const Vec3d a_lifted_B = lift(a_sphere, mesh_B_idx);
        const Vec3d b_lifted_B = lift(b_sphere, mesh_B_idx);
        const Vec3d c_lifted_B = lift(c_sphere, mesh_B_idx);

        Eigen::Vector2d a_local_A, b_local_A, c_local_A, a_local_B, b_local_B, c_local_B;
        to_local_coordinates(a_lifted_A, b_lifted_A, c_lifted_A, a_local_A, b_local_A, c_local_A);
        to_local_coordinates(a_lifted_B, b_lifted_B, c_lifted_B, a_local_B, b_local_B, c_local_B);

        Eigen::Matrix2d M_A;
        M_A << b_local_A - a_local_A, c_local_A - a_local_A;
        Eigen::Matrix2d M_B;
        M_B << b_local_B - a_local_B, c_local_B - a_local_B;

        if (M_A.determinant() <= 0 || M_B.determinant() <= 0)
            continue;

        const Eigen::Matrix2d J_actual = M_B * M_A.inverse();

        // Metric form of the prescription: only V and sigma, U is irrelevant
        Eigen::Matrix2d V_local;
        Eigen::Vector2d sigma;
        lookup_prescribed_metric(a_sphere, b_sphere, c_sphere,
                                 a_lifted_A, b_lifted_A, c_lifted_A,
                                 0, _map_state, V_local, sigma);

        if (sigma.minCoeff() <= 1e-6)
            continue;

        // U drops out of singular values, so J * V * Sigma^-1 has the same spectrum
        // as the full residual J * J*^{-1}.
        const Eigen::Matrix2d residual =
                J_actual * V_local * Eigen::Vector2d(1.0 / sigma[0], 1.0 / sigma[1]).asDiagonal();
        const Eigen::Vector2d s = Eigen::JacobiSVD<Eigen::Matrix2d>(residual).singularValues();

        // Both determinants are > 0 and sigma is bounded away from 0 (checked above), so
        // the residual is nonsingular; the clamp only guards against roundoff in the log.
        const double s0 = s[0];
        const double s1 = std::max(s[1], 1e-12);

        // det(M_A) = 2 * area of the lifted A triangle, i.e. the amount of mesh-A
        // material this T face is responsible for.
        const double area_A = 0.5 * M_A.determinant();

        devs.emplace_back((std::abs(s0 - 1.0) + std::abs(s1 - 1.0)) / 2.0, area_A);
        total_area += area_A;

        _stats.max = std::max(_stats.max, std::max(std::abs(s0 - 1.0), std::abs(s1 - 1.0)));
        _stats.area_part_aw  += area_A * std::abs(std::log(s0 * s1));
        _stats.shape_part_aw += area_A * std::abs(std::log(s0 / s1));
    }

    if (devs.empty())
        return;

    const int n = (int)devs.size();
    _stats.n_faces = n;

    for (const auto& vw : devs)
    {
        _stats.mean += vw.first;
        _stats.mean_aw += vw.second * vw.first;
        if (vw.first > 0.1)
        {
            _stats.frac_gt_0p1 += 1.0;
            _stats.frac_gt_0p1_aw += vw.second;
        }
    }
    _stats.mean /= (double)n;
    _stats.frac_gt_0p1 /= (double)n;

    if (total_area > 0.0)
    {
        _stats.mean_aw /= total_area;
        _stats.frac_gt_0p1_aw /= total_area;
        _stats.area_part_aw /= total_area;
        _stats.shape_part_aw /= total_area;
    }

    std::sort(devs.begin(), devs.end()); // by deviation
    _stats.median = devs[n / 2].first;
    _stats.p90 = devs[std::min(n - 1, (int)(0.9 * n))].first;
    _stats.median_aw = weighted_percentile(devs, total_area, 0.5);
}

/// Report both metrics at a pipeline checkpoint, and append a row to the CSV.
/// Measuring at every stage shows exactly WHERE the map drifts off ground truth.
void report(
        const MapState& _map_state,
        const std::string& _config_name,
        const std::string& _stage,
        std::ofstream& _csv)
{
    double corr_mean, corr_median, corr_max;
    compute_correspondence_error(_map_state, corr_mean, corr_median, corr_max);

    ResidualStats res;
    compute_metric_residual(_map_state, res);

    ISM_INFO("[" << _config_name << " | " << _stage << "] "
             << "corr_err mean=" << corr_mean << " median=" << corr_median << " max=" << corr_max
             << " | residual mean=" << res.mean << " (aw " << res.mean_aw << ")"
             << " median=" << res.median << " p90=" << res.p90 << " max=" << res.max
             << " frac>0.1=" << res.frac_gt_0p1
             << " | area=" << res.area_part_aw << " shape=" << res.shape_part_aw
             << " (" << res.n_faces << " faces)"
             << " | |V(T)|=" << _map_state.mesh_T.n_vertices());

    _csv << _config_name << "," << _stage << ","
         << corr_mean << "," << corr_median << "," << corr_max << ","
         << res.mean << "," << res.mean_aw << ","
         << res.median << "," << res.median_aw << "," << res.p90 << "," << res.max << ","
         << res.frac_gt_0p1 << "," << res.frac_gt_0p1_aw << ","
         << res.area_part_aw << "," << res.shape_part_aw << ","
         << res.n_faces << "," << _map_state.mesh_T.n_vertices() << "\n";
    _csv.flush();
}

/// Zero-padded frame index, so the dumped meshes sort in optimization order.
std::string pad4(const int _i)
{
    const std::string s = std::to_string(_i);
    return std::string(std::max(0, 4 - (int)s.size()), '0') + s;
}

/// One row of the per-iteration trace, plus (optionally) a mesh pair for the animation.
/// Called from the optimizer's own callbacks, so the state is a completed iteration:
/// embeddings_T are updated, the local bases are re-centered, and the vertex-to-T-face
/// assignments were refreshed inside the line search.
void trace_step(
        const MapState& _map_state,
        const fs::path& _output_dir,
        const std::string& _kind,
        const std::string& _label,
        int& _frame,
        std::ofstream& _csv)
{
    double corr_mean, corr_median, corr_max;
    compute_correspondence_error(_map_state, corr_mean, corr_median, corr_max);

    ResidualStats res;
    compute_metric_residual(_map_state, res);

    // Area-weighted only: T changes size constantly here, so the unweighted numbers
    // would not be comparable from one row to the next.
    _csv << _frame << "," << _kind << "," << _label << ","
         << corr_mean << "," << corr_median << ","
         << res.mean_aw << "," << res.area_part_aw << "," << res.shape_part_aw << ","
         << res.frac_gt_0p1_aw << "," << res.n_faces << ","
         << _map_state.mesh_T.n_vertices() << "\n";
    _csv.flush();

    if (TRACE_WRITE_MESHES && (_frame % std::max(1, TRACE_MESH_STRIDE) == 0))
    {
        const std::vector<TriMesh> lifted = lifted_meshes_from_mapstate(_map_state);
        const fs::path dir = _output_dir / "trace";
        write_mesh(lifted[0], dir / ("f" + pad4(_frame) + "_" + _kind + "_A.obj"));
        write_mesh(lifted[1], dir / ("f" + pad4(_frame) + "_" + _kind + "_B.obj"));
    }

    ++_frame;
}

// ---------------------------------------------------------------------------
// One configuration
// ---------------------------------------------------------------------------

void run_config(
        const Config& _config,
        const fs::path& _output_root,
        std::ofstream& _csv,
        std::vector<TriMesh>& _out_lifted_A,
        std::vector<TriMesh>& _out_lifted_B,
        std::vector<MapState>& _out_states)
{
    const std::string name = _config.name();
    ISM_INFO("################ " << name << " ################");

    const fs::path output_dir = _output_root / name;
    fs::create_directories(output_dir);

    // A on A maps the mesh to itself: both inputs are the same file.
    const fs::path mesh_path_A = DATA_PATH / "meshes/l_shape/newL.obj";
    const fs::path mesh_path_B = (_config.pair_mode == PairMode::SelfMap)
            ? mesh_path_A
            : DATA_PATH / "meshes/l_shape/stretchedL.obj";

    // --- Landmarks -------------------------------------------------------
    // Written into the OUTPUT dir (never into DATA_PATH). The two meshes share
    // connectivity, so identical vertex indices are valid ground-truth pairs.
    const fs::path landmarks_path_A = output_dir / "landmarks_A.pinned";
    const fs::path landmarks_path_B = output_dir / "landmarks_B.pinned";
    {
        const TriMesh mesh_A = read_mesh(mesh_path_A);
        std::vector<VH> landmarks;
        if (USE_FPS_LANDMARKS)
        {
            landmarks = farthest_point_sampling(mesh_A, N_FPS_LANDMARKS);
            ISM_INFO("FPS landmarks (" << landmarks.size() << "): ");
        }
        else
        {
            for (int i = 0; i < 3; ++i)
                landmarks.push_back(VH(i));
            ISM_INFO("Default landmarks (first 3 vertices)");
        }
        write_landmarks_pinned(landmarks, landmarks_path_A, {}, true);
        write_landmarks_pinned(landmarks, landmarks_path_B, {}, true);
    }

    // --- Sphere embeddings ------------------------------------------------
    // The embedding depends on the landmarks, so each config needs its own cache
    // path -- reusing a stale embedding would silently invalidate the comparison.
    const fs::path embedding_path_A = output_dir / "embedding_A.obj";
    const fs::path embedding_path_B = output_dir / "embedding_B.obj";

    // Share one embedding between both meshes when:
    //  - SelfMap: the two inputs ARE the same mesh, so init_map computing two
    //    embeddings independently could yield different ones and the initial map
    //    would not be the identity -- which is exactly what this test must check.
    //  - Stretched + START_FROM_GROUND_TRUTH: the optional ablation.
    const bool share_embedding = (_config.pair_mode == PairMode::SelfMap)
            || START_FROM_GROUND_TRUTH;

    if (share_embedding)
    {
        // Compute mesh A's embedding and use it for BOTH meshes, so the initial map
        // is exactly the ground-truth correspondence (the identity, for SelfMap).
        // init_map only computes an embedding when the file is absent, so writing
        // both files here makes it load ours (and skip its rotation alignment).
        //
        // Cache the embedding ONCE at the output root, then copy it into this config's
        // paths on EVERY run -- never trust the per-config files to have been written
        // with the current settings.
        const fs::path shared_cache = _output_root / "embedding_shared.obj";
        if (!fs::exists(shared_cache))
        {
            TriMesh mesh_A = read_mesh(mesh_path_A);
            center_mesh(mesh_A);
            normalize_surface_area(mesh_A);

            const ExternalProperty<VH, Vec3d> embedding = multi_res_sphere_embedding(mesh_A);
            ISM_ASSERT(sphere_embedding_bijective(mesh_A, embedding));

            write_embedding(mesh_A, embedding, shared_cache);
        }
        fs::copy_file(shared_cache, embedding_path_A, fs::copy_options::overwrite_existing);
        fs::copy_file(shared_cache, embedding_path_B, fs::copy_options::overwrite_existing);
        ISM_INFO("Using shared sphere embedding (initial map = ground truth)");
    }

    if (_config.pair_mode == PairMode::SelfMap)
        ISM_INFO("Self-map consistency check: expecting the identity map");

    // --- Init -------------------------------------------------------------
    MapState map_state;
    init_map(map_state,
             { mesh_path_A, mesh_path_B },
             { landmarks_path_A, landmarks_path_B },
             { embedding_path_A, embedding_path_B },
             false);

    // Verify the shared embedding actually took effect. Without this, a stale cached
    // embedding silently turns the ablation into a no-op and the run looks normal.
    if (share_embedding)
    {
        double max_diff = 0.0;
        for (auto vh : map_state.meshes_input[0].vertices())
            max_diff = std::max(max_diff,
                    (map_state.embeddings_input[0][vh] - map_state.embeddings_input[1][vh]).norm());
        ISM_INFO("Shared embedding check: max |emb_A - emb_B| = " << max_diff);
        ISM_ASSERT_L(max_diff, 1e-12);
    }

    map_state.set_distortion_pairs(DistortionPairs::All);

    // Extract the prescribed field for ALL configs: the Dirichlet baselines do not
    // optimize it, but they are still evaluated against it.
    map_state.prescribed_jacobians.push_back(
                extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]));

    assign_vertices_to_T_faces(map_state);

    report(map_state, name, "00_init", _csv);

    // --- Per-iteration trace ----------------------------------------------
    // tracer(kind) builds a callback for one call site; the optimizer supplies the
    // label ("after_optimization", "after_splits", ...). Everything is captured by
    // reference except the kind, which is copied into the callback.
    std::ofstream trace_csv;
    int trace_frame = 0;
    if (TRACE_ITERATIONS)
    {
        fs::create_directories(output_dir / "trace");
        trace_csv.open(output_dir / "trace.csv");
        trace_csv << "frame,kind,label,corr_mean,corr_median,"
                     "residual_mean_aw,residual_area_aw,residual_shape_aw,"
                     "residual_frac_gt_0p1_aw,n_faces_T,n_verts_T\n";
    }
    auto tracer = [&] (const std::string& _kind)
    {
        return [&, _kind] (const std::string& _label)
        {
            if (TRACE_ITERATIONS)
                trace_step(map_state, output_dir, _kind, _label, trace_frame, trace_csv);
        };
    };

    // --- Pipeline (identical across configs; only the energy differs) ------
    TinyAD::Timer timer(name);

    landmark_phase(map_state);
    report(map_state, name, "01_landmark", _csv);

    const bool prescribed = (_config.energy_mode == EnergyMode::PrescribedMetric);

    // Inlined coarse_phase(): that helper builds its own settings and accepts no
    // overrides, so we replicate its two lines here to be able to change them.
    {
        ISM_INFO("#################### Starting Coarse Phase ####################");
        AdaptiveTriangulationsSettings settings = coarse_phase_settings();
        settings.use_prescribed_jacobian = prescribed && PRESCRIBED_IN_COARSE_PHASE;
        settings.prescribed_metric_form = true;
        settings.w_map = W_MAP;
        if (DISABLE_REMESHING)
            settings.allow_splits = settings.allow_collapses = settings.allow_flips = false;
        optimize_with_remeshing(map_state, settings, "",
                                tracer("coarse_optim"), tracer("coarse_remesh"));
    }
    report(map_state, name, "02_coarse", _csv);

    {
        AdaptiveTriangulationsSettings settings = fine_phase_settings();
        settings.use_prescribed_jacobian = prescribed;
        settings.prescribed_metric_form = true; // metric form, not the legacy full-J* residual
        settings.w_map = W_MAP;
        settings.max_iterations = 50;
        if (DISABLE_REMESHING)
            settings.allow_splits = settings.allow_collapses = settings.allow_flips = false;
        optimize_with_remeshing(map_state, settings, "",
                                tracer("fine_optim"), tracer("fine_remesh"));
    }
    report(map_state, name, "03_final", _csv);

    timer.stop();
    ISM_INFO("[" << name << "] runtime " << timer.seconds() << " s");

    // --- Output -----------------------------------------------------------
    std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(map_state);
    for (int i = 0; i < (int)map_state.meshes_input.size(); ++i)
        write_mesh(lifted_Ts[i], output_dir / ("T_lifted_" + std::to_string(i) + ".obj"));

    _out_lifted_A.push_back(lifted_Ts[0]);
    _out_lifted_B.push_back(lifted_Ts[1]);
    _out_states.push_back(map_state);
}

// ---------------------------------------------------------------------------

void run()
{
    const fs::path output_root = OUTPUT_PATH / "baseline_comparison";
    const fs::path screenshot_dir = output_root / "screenshots";
    fs::create_directories(screenshot_dir);

    glow::SharedTexture2D texture = read_texture(DATA_PATH / "textures/checkerboard.png");

    // CSV of all metrics, one row per (config, stage)
    const std::string suffix = (USE_FPS_LANDMARKS ? ("fps" + std::to_string(N_FPS_LANDMARKS)) : "default3")
            + std::string(START_FROM_GROUND_TRUTH ? "_gt" : "")
            + std::string(PRESCRIBED_IN_COARSE_PHASE ? "_pcoarse" : "")
            + std::string(DISABLE_REMESHING ? "_noremesh" : "")
            + std::string(W_MAP == 1.0 ? "" : ("_wmap" + std::to_string((int)W_MAP)));
    std::ofstream csv(output_root / ("metrics_" + suffix + ".csv"));
    csv << "config,stage,corr_mean,corr_median,corr_max,"
           "residual_mean,residual_mean_aw,residual_median,residual_median_aw,"
           "residual_p90,residual_max,residual_frac_gt_0p1,residual_frac_gt_0p1_aw,"
           "residual_area_aw,residual_shape_aw,"
           "n_faces_T,n_verts_T\n";

    const std::vector<Config> configs = {
        { PairMode::SelfMap,   EnergyMode::SymmetricDirichlet },
        { PairMode::SelfMap,   EnergyMode::PrescribedMetric   },
        { PairMode::Stretched, EnergyMode::SymmetricDirichlet },
        { PairMode::Stretched, EnergyMode::PrescribedMetric   },
    };

    std::vector<TriMesh> lifted_As, lifted_Bs;
    std::vector<MapState> states;
    for (const auto& config : configs)
        run_config(config, output_root, csv, lifted_As, lifted_Bs, states);

    csv.close();
    ISM_INFO("Wrote metrics_" << suffix << ".csv");

    // --- Visualization: 2 per config = 8 total ---------------------------
    // Row 1: geometry + wireframe on B (does T look sane, do triangles slide?)
    // Row 2: checkerboard transferred A -> B (sliding and shear are obvious here)
    for (int i = 0; i < (int)configs.size(); ++i)
    {
        const std::string name = configs[i].name();

        {
            auto s = screenshot_config(OPEN_VIEWER, screenshot_dir / (name + "_wire.png"), tg::ivec2(1920, 1080), true);
            auto style = default_style();
            auto g = gv::grid();
            {
                auto v = gv::view();
                view_mesh(states[i].meshes_input[0], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_As[i], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_As[i], MAGENTA, WidthScreen(0.5));
            }
            {
                auto v = gv::view();
                view_mesh(states[i].meshes_input[1], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_Bs[i], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_Bs[i], TEAL, WidthScreen(0.5));
            }
        }

        {
            auto s = screenshot_config(OPEN_VIEWER, screenshot_dir / (name + "_texture.png"), tg::ivec2(1920, 1080), true);
            auto style = default_style();
            auto g = gv::grid();
            {
                auto v = gv::view();
                view_texture_frontal_projection_input(states[i], 0, 0, TEXTURE_PROJECTION_DIR, TEXTURE_FACTOR, texture);
            }
            {
                auto v = gv::view();
                view_texture_frontal_projection_input(states[i], 0, 1, TEXTURE_PROJECTION_DIR, TEXTURE_FACTOR, texture);
            }
        }
    }

    ISM_INFO("Done.");
}

}  // namespace SurfaceMaps

int main()
{
    glow::glfw::GlfwContext ctx;
    using namespace SurfaceMaps;
    init_lib_surface_maps();
    run();
    return 0;
}
