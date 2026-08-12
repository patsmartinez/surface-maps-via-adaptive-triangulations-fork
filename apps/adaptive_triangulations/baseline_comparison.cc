/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Baseline comparison for the prescribed Jacobian work.
 *
 * Runs 4 configurations over the L-shape test pair:
 *
 *                   | Symmetric Dirichlet | Prescribed metric
 *   ----------------+---------------------+-------------------
 *   Shared emb.     |  "A on A" baseline  |  "A on A" ablation
 *   Independent emb.|  "A on B" baseline  |  "A on B" (the real run)
 *
 * "Shared embedding" writes mesh A's sphere embedding for BOTH meshes. Since the
 * two L meshes share connectivity, the initial map is then exactly the ground-truth
 * correspondence -- so the question becomes whether the energy HOLDS the map there.
 * This isolates energy correctness from initialization quality.
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

// ---------------------------------------------------------------------------

enum class EmbeddingMode
{
    Shared,      // "A on A": mesh A's embedding used for both -> init map is ground truth
    Independent, // "A on B": each mesh embedded on its own (the normal pipeline)
};

enum class EnergyMode
{
    SymmetricDirichlet, // Original algorithm
    PrescribedMetric,   // Metric-form prescribed Jacobian
};

struct Config
{
    EmbeddingMode embedding_mode;
    EnergyMode energy_mode;

    std::string name() const
    {
        const std::string emb = (embedding_mode == EmbeddingMode::Shared) ? "AonA" : "AonB";
        const std::string energy = (energy_mode == EnergyMode::SymmetricDirichlet) ? "dirichlet" : "prescribed";
        const std::string lm = USE_FPS_LANDMARKS ? ("fps" + std::to_string(N_FPS_LANDMARKS)) : "default3";
        return emb + "_" + energy + "_" + lm;
    }
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Ground-truth correspondence error.
/// The two L meshes share connectivity, so the true image of vertex v of mesh A is
/// vertex v of mesh B. Errors are normalized by sqrt(area of B) so they are
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

/// Rotation-invariant residual against the prescribed field: singular values of
/// J * J*^{-1}, which are all 1 exactly when the pullback metric matches the
/// prescription. Computed for ALL configs (the baselines need it for evaluation
/// even though they do not optimize it).
void compute_metric_residual(
        const MapState& _map_state,
        double& _mean_deviation,
        double& _max_deviation)
{
    _mean_deviation = 0.0;
    _max_deviation = 0.0;
    int count = 0;

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

        const double dev = std::max(std::abs(s[0] - 1.0), std::abs(s[1] - 1.0));
        _mean_deviation += (std::abs(s[0] - 1.0) + std::abs(s[1] - 1.0)) / 2.0;
        _max_deviation = std::max(_max_deviation, dev);
        count++;
    }

    if (count > 0)
        _mean_deviation /= (double)count;
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

    double res_mean, res_max;
    compute_metric_residual(_map_state, res_mean, res_max);

    ISM_INFO("[" << _config_name << " | " << _stage << "] "
             << "corr_err mean=" << corr_mean << " median=" << corr_median << " max=" << corr_max
             << " | residual mean=" << res_mean << " max=" << res_max
             << " | |V(T)|=" << _map_state.mesh_T.n_vertices());

    _csv << _config_name << "," << _stage << ","
         << corr_mean << "," << corr_median << "," << corr_max << ","
         << res_mean << "," << res_max << ","
         << _map_state.mesh_T.n_vertices() << "\n";
    _csv.flush();
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

    const fs::path mesh_path_A = DATA_PATH / "meshes/l_shape/newL.obj";
    const fs::path mesh_path_B = DATA_PATH / "meshes/l_shape/stretchedL.obj";

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

    if (_config.embedding_mode == EmbeddingMode::Shared)
    {
        // "A on A": compute mesh A's embedding and use it for BOTH meshes.
        // init_map only computes an embedding when the file is absent, so writing
        // both files here makes it load ours (and skip its rotation alignment).
        if (!fs::exists(embedding_path_A) || !fs::exists(embedding_path_B))
        {
            TriMesh mesh_A = read_mesh(mesh_path_A);
            center_mesh(mesh_A);
            normalize_surface_area(mesh_A);

            const ExternalProperty<VH, Vec3d> embedding = multi_res_sphere_embedding(mesh_A);
            ISM_ASSERT(sphere_embedding_bijective(mesh_A, embedding));

            write_embedding(mesh_A, embedding, embedding_path_A);
            write_embedding(mesh_A, embedding, embedding_path_B);
            ISM_INFO("Wrote shared sphere embedding (initial map = ground truth)");
        }
    }

    // --- Init -------------------------------------------------------------
    MapState map_state;
    init_map(map_state,
             { mesh_path_A, mesh_path_B },
             { landmarks_path_A, landmarks_path_B },
             { embedding_path_A, embedding_path_B },
             false);

    map_state.set_distortion_pairs(DistortionPairs::All);

    // Extract the prescribed field for ALL configs: the Dirichlet baselines do not
    // optimize it, but they are still evaluated against it.
    map_state.prescribed_jacobians.push_back(
                extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]));

    assign_vertices_to_T_faces(map_state);

    report(map_state, name, "00_init", _csv);

    // --- Pipeline (identical across configs; only the energy differs) ------
    TinyAD::Timer timer(name);

    landmark_phase(map_state);
    report(map_state, name, "01_landmark", _csv);

    coarse_phase(map_state);
    report(map_state, name, "02_coarse", _csv);

    AdaptiveTriangulationsSettings settings = fine_phase_settings();
    settings.use_prescribed_jacobian = (_config.energy_mode == EnergyMode::PrescribedMetric);
    settings.max_iterations = 50;
    optimize_with_remeshing(map_state, settings);
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
    const std::string suffix = USE_FPS_LANDMARKS ? ("fps" + std::to_string(N_FPS_LANDMARKS)) : "default3";
    std::ofstream csv(output_root / ("metrics_" + suffix + ".csv"));
    csv << "config,stage,corr_mean,corr_median,corr_max,residual_mean,residual_max,n_verts_T\n";

    const std::vector<Config> configs = {
        { EmbeddingMode::Shared,      EnergyMode::SymmetricDirichlet },
        { EmbeddingMode::Shared,      EnergyMode::PrescribedMetric   },
        { EmbeddingMode::Independent, EnergyMode::SymmetricDirichlet },
        { EmbeddingMode::Independent, EnergyMode::PrescribedMetric   },
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
                view_texture_frontal_projection_input(states[i], 0, 0, 1, 1.0, texture);
            }
            {
                auto v = gv::view();
                view_texture_frontal_projection_input(states[i], 0, 1, 1, 1.0, texture);
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
