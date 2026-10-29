/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * LOOKUP DISTORTION PROBE -- is the J* lookup error driven by sphere-embedding
 * distortion?
 *
 * Background. Holding the map EXACTLY at the ground truth and refining T (see
 * baseline_comparison's REMESH_ONLY), the residual stops being map error and becomes
 * purely "how wrong is the J* the energy looks up on this T triangle". Measured
 * 2026-10-28, area-weighted, at matched |F(T)|/|F(input)|:
 *
 *     L       0.0067 .. 0.0145   over ratio 0.21 .. 2.68
 *     sphere  0.0066 .. 0.0144   over ratio 2.14 .. 34.1
 *     U       0.0463 .. 0.0722   over ratio 0.046 .. 0.73
 *
 * So the U is 7-9x worse than the other two at matched resolution, AND the error does
 * not shrink as T refines. That rules out intra-triangle aliasing -- a point sample of a
 * field that varies ACROSS a triangle must improve as the triangle shrinks -- so
 * averaging more samples inside the triangle would not fix it.
 *
 * The hypothesis this probe tests. The lookup samples at the T triangle's centroid
 * ON THE SPHERE and finds the containing face of mesh A's EMBEDDING
 * (PrescribedJacobianField.cc, lookup_prescribed_metric). T, however, is sized by the
 * SURFACE curvature field. Where the sphere embedding is strongly area-distorted those
 * two disagree: a T triangle that is small on the surface can still straddle a large,
 * rapidly-varying patch in sphere space, and refining T uniformly in surface terms never
 * preferentially refines exactly those places. That would produce a resolution-
 * independent error concentrated on the distorted regions -- which is what we see.
 *
 * What it measures. At the frozen ground-truth configuration, per T face:
 *   err         the same per-face deviation compute_metric_residual accumulates,
 *               (|s0-1| + |s1-1|)/2 for the singular values of J J*^-1
 *   distortion  |log(mu_f / mu_mean)| of the input face the lookup actually landed in,
 *               where mu_f = (area of f in the sphere embedding) / (area of f on the
 *               surface). Zero means that face is carried to the sphere with the average
 *               area scaling; large means it is locally compressed or expanded.
 * then correlates them (Pearson on the raw pair and on logs, plus Spearman, plus a
 * decile table, because a single coefficient hides a threshold effect).
 *
 * How to read it:
 *   strong positive correlation, U's high-error faces are its high-distortion faces
 *       -> confirmed. Size T by the variation of J*, not by surface curvature.
 *   no correlation
 *       -> refuted. The error is elsewhere: the V-frame projection, the BSP lookup
 *          itself, or J* being unachievable on the U.
 *
 * Runs no optimization: landmark phase, then remesh-only coarse and fine passes. The
 * printed corr_mean must stay at ~1e-17 or the measurement is void.
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
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeWithRemeshing.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/MultiRes/MultiResSphereEmbedding.hh>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

namespace SurfaceMaps
{
namespace
{

std::string MESH_DIR = "meshes/u_shape";
std::string MESH_A   = "u1.obj";
std::string MESH_B   = "u1stretch.obj";
int    N_FPS_LANDMARKS   = 16;
double APPROX_ERROR_SCALE = 1.0;
double W_MESH            = 0.01;

/// Cap on the sizing field, overriding the phases' default of 100.0 (i.e. no cap at all
/// on a mesh normalized to surface area 1).
///
/// This is the knob APPROX_ERROR_SCALE could not reach. tel = sqrt(6*e/kappa - 3*e^2)
/// CLAMPED to [min_edge_length, max_edge_length]: on a flat region kappa -> 0 sends tel
/// to infinity and the clamp pins it at max_edge_length no matter what e is. So scaling
/// the approximation error refines T only where it is already curved, and leaves the
/// flat regions exactly as coarse as they were -- which is where the lookup error turns
/// out to live.
double MAX_EDGE_LENGTH   = 100.0;

/// Per-face area scaling of the sphere embedding, as |log(mu_f / mu_mean)|.
///
/// mu_f = embedding-triangle area / surface-triangle area. Dividing by the global mean
/// makes it scale-free, so 0 means "this face is carried to the sphere with the average
/// area scaling" regardless of how the mesh or the sphere is normalized, and the measure
/// is symmetric between compression and expansion.
ExternalProperty<FH, double> embedding_area_distortion(
        const TriMesh& _mesh,
        const TriMesh& _embedding)
{
    ExternalProperty<FH, double> mu(_mesh, 0.0);
    double sum_surf = 0.0, sum_emb = 0.0;

    std::vector<double> raw(_mesh.n_faces(), 0.0);
    for (auto fh : _mesh.faces())
    {
        const double a_surf = _mesh.calc_face_area(fh);
        const double a_emb  = _embedding.calc_face_area(fh);
        sum_surf += a_surf;
        sum_emb  += a_emb;
        raw[fh.idx()] = (a_surf > 1e-30) ? (a_emb / a_surf) : 0.0;
    }
    const double mu_mean = (sum_surf > 0.0) ? (sum_emb / sum_surf) : 1.0;

    for (auto fh : _mesh.faces())
    {
        const double r = raw[fh.idx()];
        mu[fh] = (r > 1e-30 && mu_mean > 0.0) ? std::abs(std::log(r / mu_mean)) : 0.0;
    }
    return mu;
}

/// Ground-truth correspondence error, same definition baseline_comparison uses: the two
/// meshes share connectivity, so vertex v of A must land on vertex v of B; normalized by
/// sqrt(area of B). Here it is purely a validity check -- with remesh_only it must stay
/// at ~1e-17, or the map moved and the lookup numbers mean nothing.
double correspondence_error_mean(
        const MapState& _map_state)
{
    const TriMesh mapped_A = map_vertices_to_target(_map_state, 0, 1);
    const TriMesh& mesh_B = _map_state.meshes_input[1];
    ISM_ASSERT_EQ(mapped_A.n_vertices(), mesh_B.n_vertices());

    double area_B = 0.0;
    for (auto fh : mesh_B.faces())
        area_B += mesh_B.calc_face_area(fh);
    const double scale = 1.0 / std::sqrt(area_B);

    double sum = 0.0;
    for (auto vh : mesh_B.vertices())
        sum += (mapped_A.point(vh) - mesh_B.point(vh)).norm() * scale;
    return sum / (double)mesh_B.n_vertices();
}

double pearson(const std::vector<double>& _x, const std::vector<double>& _y)
{
    const int n = (int)_x.size();
    if (n < 3) return std::nan("");
    const double mx = std::accumulate(_x.begin(), _x.end(), 0.0) / n;
    const double my = std::accumulate(_y.begin(), _y.end(), 0.0) / n;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double dx = _x[i] - mx, dy = _y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return (sxx > 0.0 && syy > 0.0) ? sxy / std::sqrt(sxx * syy) : std::nan("");
}

std::vector<double> ranks(const std::vector<double>& _v)
{
    const int n = (int)_v.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return _v[a] < _v[b]; });
    std::vector<double> r(n);
    int i = 0;
    while (i < n) // average ties, or the coefficient is biased when many faces share a value
    {
        int j = i;
        while (j + 1 < n && _v[idx[j + 1]] == _v[idx[i]]) ++j;
        const double avg = 0.5 * (i + j) + 1.0;
        for (int k = i; k <= j; ++k) r[idx[k]] = avg;
        i = j + 1;
    }
    return r;
}

void run()
{
    const fs::path mesh_path_A = DATA_PATH / MESH_DIR / MESH_A;
    const fs::path mesh_path_B = DATA_PATH / MESH_DIR / MESH_B;
    const std::string shape = fs::path(MESH_DIR).filename().string();

    const fs::path out_dir = OUTPUT_PATH / ("lookup_distortion_" + shape);
    fs::create_directories(out_dir);

    // --- Landmarks --------------------------------------------------------
    const fs::path lm_A = out_dir / "landmarks_A.pinned";
    const fs::path lm_B = out_dir / "landmarks_B.pinned";
    {
        const TriMesh mesh_A = read_mesh(mesh_path_A);
        const std::vector<VH> landmarks = farthest_point_sampling(mesh_A, N_FPS_LANDMARKS);
        write_landmarks_pinned(landmarks, lm_A, {}, true);
        write_landmarks_pinned(landmarks, lm_B, {}, true);
    }

    // --- Shared embedding = ground-truth initial map ----------------------
    // Reuse baseline_comparison's cache when it exists, so this probe measures the SAME
    // embedding the baselines ran on rather than an independently recomputed one.
    const fs::path emb_A = out_dir / "embedding_A.obj";
    const fs::path emb_B = out_dir / "embedding_B.obj";
    const fs::path shared_cache =
            OUTPUT_PATH / ("baseline_comparison_" + shape) / "embedding_shared.obj";
    if (fs::exists(shared_cache))
    {
        ISM_INFO("Reusing baseline_comparison's shared embedding");
    }
    else
    {
        TriMesh mesh_A = read_mesh(mesh_path_A);
        center_mesh(mesh_A);
        normalize_surface_area(mesh_A);
        const ExternalProperty<VH, Vec3d> e = multi_res_sphere_embedding(mesh_A);
        ISM_ASSERT(sphere_embedding_bijective(mesh_A, e));
        fs::create_directories(shared_cache.parent_path());
        write_embedding(mesh_A, e, shared_cache);
    }
    fs::copy_file(shared_cache, emb_A, fs::copy_options::overwrite_existing);
    fs::copy_file(shared_cache, emb_B, fs::copy_options::overwrite_existing);

    // --- Init -------------------------------------------------------------
    MapState map_state;
    init_map(map_state, { mesh_path_A, mesh_path_B }, { lm_A, lm_B }, { emb_A, emb_B }, false);
    {
        double d = 0.0;
        for (auto vh : map_state.meshes_input[0].vertices())
            d = std::max(d, (map_state.embeddings_input[0][vh] - map_state.embeddings_input[1][vh]).norm());
        ISM_ASSERT_L(d, 1e-12); // without this a stale cache silently voids the GT start
    }
    map_state.set_distortion_pairs(DistortionPairs::All);
    map_state.prescribed_jacobians.push_back(
                extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]));

    // --- Refine T without ever moving the map -----------------------------
    landmark_phase(map_state);
    for (int phase = 0; phase < 2; ++phase)
    {
        AdaptiveTriangulationsSettings s =
                (phase == 0) ? coarse_phase_settings() : fine_phase_settings();
        s.use_prescribed_jacobian = true;
        s.prescribed_metric_form = true;
        s.mesh_tel_passive = true;
        s.remesh_only = true;              // remesh, never step
        s.approx_error *= APPROX_ERROR_SCALE;
        s.max_edge_length = std::min(s.max_edge_length, MAX_EDGE_LENGTH);
        s.w_mesh = W_MESH;
        s.max_iterations = 50;
        optimize_with_remeshing(map_state, s, "");
    }

    ISM_HIGHLIGHT("VALIDITY: corr_mean = " << correspondence_error_mean(map_state)
                  << " (must be ~1e-17; the map must not have moved)");

    // --- Per-face measurement ---------------------------------------------
    const int idx_A = map_state.pairs_map_distortion[0].first;
    const int idx_B = map_state.pairs_map_distortion[0].second;
    const ExternalProperty<FH, double> mu =
            embedding_area_distortion(map_state.meshes_input[idx_A],
                                      map_state.meshes_embeddings_input[idx_A]);

    std::ofstream csv(out_dir / ("per_face_approx" + std::to_string(APPROX_ERROR_SCALE)
                                + "_mel" + std::to_string(MAX_EDGE_LENGTH) + ".csv"));
    csv << "t_face,area_A,err,err_area,err_shape,distortion,input_face\n";

    std::vector<double> errs, dists, areas;
    int skipped = 0;
    for (auto fh : map_state.mesh_T.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(map_state.mesh_T, fh, vh_a, vh_b, vh_c);

        const Vec3d a_s = map_state.embeddings_T[idx_A][vh_a];
        const Vec3d b_s = map_state.embeddings_T[idx_A][vh_b];
        const Vec3d c_s = map_state.embeddings_T[idx_A][vh_c];

        auto lift = [&](const Vec3d& p, int i) {
            return lift_vertex_to_surface(p, map_state.meshes_input[i],
                    map_state.meshes_embeddings_input[i], map_state.bsp_embeddings_input[i]);
        };
        const Vec3d aA = lift(a_s, idx_A), bA = lift(b_s, idx_A), cA = lift(c_s, idx_A);
        const Vec3d aB = lift(map_state.embeddings_T[idx_B][vh_a], idx_B);
        const Vec3d bB = lift(map_state.embeddings_T[idx_B][vh_b], idx_B);
        const Vec3d cB = lift(map_state.embeddings_T[idx_B][vh_c], idx_B);

        Eigen::Vector2d a2A, b2A, c2A, a2B, b2B, c2B;
        to_local_coordinates(aA, bA, cA, a2A, b2A, c2A);
        to_local_coordinates(aB, bB, cB, a2B, b2B, c2B);
        Eigen::Matrix2d M_A; M_A << b2A - a2A, c2A - a2A;
        Eigen::Matrix2d M_B; M_B << b2B - a2B, c2B - a2B;
        if (M_A.determinant() <= 0 || M_B.determinant() <= 0) { ++skipped; continue; }

        const Eigen::Matrix2d J = M_B * M_A.inverse();

        Eigen::Matrix2d V_local;
        Eigen::Vector2d sigma;
        lookup_prescribed_metric(a_s, b_s, c_s, aA, bA, cA, 0, map_state, V_local, sigma);
        if (sigma.minCoeff() <= 1e-6) { ++skipped; continue; }

        const Eigen::Matrix2d R =
                J * V_local * Eigen::Vector2d(1.0 / sigma[0], 1.0 / sigma[1]).asDiagonal();
        const Eigen::Vector2d s = Eigen::JacobiSVD<Eigen::Matrix2d>(R).singularValues();
        const double s0 = s[0], s1 = std::max(s[1], 1e-12);

        const double err = (std::abs(s0 - 1.0) + std::abs(s1 - 1.0)) / 2.0;
        const double area_A = 0.5 * M_A.determinant();

        // The distortion AT THE SAMPLE POINT -- the same centroid and the same BSP query
        // the lookup itself performs, so the two quantities refer to the same face.
        Vec3d centroid = (a_s + b_s + c_s) / 3.0;
        centroid.normalize();
        SFH containing;
        double al, be, ga;
        bsp_tree_barys_face(centroid, map_state.meshes_embeddings_input[idx_A],
                            map_state.bsp_embeddings_input[idx_A], al, be, ga, containing);
        const double dist = mu[FH(containing.idx())];

        csv << fh.idx() << "," << area_A << "," << err << ","
            << std::abs(std::log(s0 * s1)) << "," << std::abs(std::log(s0 / s1)) << ","
            << dist << "," << containing.idx() << "\n";

        errs.push_back(err); dists.push_back(dist); areas.push_back(area_A);
    }

    ISM_HIGHLIGHT("faces measured = " << errs.size() << ", skipped = " << skipped);
    if (errs.size() < 3) { ISM_WARNING("too few faces"); return; }

    std::vector<double> log_err, log_dist;
    for (size_t i = 0; i < errs.size(); ++i)
    {
        log_err.push_back(std::log(std::max(errs[i], 1e-12)));
        log_dist.push_back(std::log(std::max(dists[i], 1e-12)));
    }
    ISM_HIGHLIGHT("pearson(err, distortion)         = " << pearson(errs, dists));
    ISM_HIGHLIGHT("pearson(log err, log distortion) = " << pearson(log_err, log_dist));
    ISM_HIGHLIGHT("spearman                         = " << pearson(ranks(errs), ranks(dists)));

    // Decile table: a single coefficient hides a threshold, and a threshold is exactly
    // what "the error lives in the badly distorted regions" would look like.
    std::vector<int> order(errs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return dists[a] < dists[b]; });
    ISM_HIGHLIGHT("decile of embedding distortion -> mean lookup error (area-weighted)");
    const int n = (int)order.size();
    for (int d = 0; d < 10; ++d)
    {
        const int lo = d * n / 10, hi = (d + 1) * n / 10;
        if (hi <= lo) continue;
        double we = 0.0, wa = 0.0, dsum = 0.0;
        for (int k = lo; k < hi; ++k)
        {
            we += areas[order[k]] * errs[order[k]];
            wa += areas[order[k]];
            dsum += dists[order[k]];
        }
        ISM_HIGHLIGHT("  decile " << d << "  distortion~" << dsum / (hi - lo)
                      << "   mean err = " << (wa > 0 ? we / wa : 0.0));
    }
}

}
}

int main(int argc, char** argv)
{
    using namespace SurfaceMaps;
    if (argc >= 4)
    {
        MESH_DIR = argv[1]; MESH_A = argv[2]; MESH_B = argv[3];
        if (argc >= 5) N_FPS_LANDMARKS = std::atoi(argv[4]);
        if (argc >= 6) APPROX_ERROR_SCALE = std::atof(argv[5]);
        if (argc >= 7) MAX_EDGE_LENGTH = std::atof(argv[6]);
    }
    else if (argc != 1)
    {
        ISM_ERROR_throw("usage: lookup_distortion_probe [mesh_dir mesh_A mesh_B [fps [approx_scale]]]");
    }
    init_lib_surface_maps();
    run();
    return 0;
}
