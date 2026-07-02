/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Test app for prescribed Jacobian optimization.
 * Uses an L-shaped mesh and its stretched version as a ground truth test case.
 */
#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/Viewer/MeshView.hh>
#include <SurfaceMaps/Viewer/HeatmapColors.hh>
#include <SurfaceMaps/Viewer/Colors.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedJacobianField.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeWithRemeshing.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Visualization.hh>

#include <Eigen/SVD>
#include <TinyAD/Utils/Timer.hh>
#include <fstream>

namespace SurfaceMaps
{

/// Compute the residual Jacobian error for verification
/// Returns the mean and max Frobenius norm of ||J - J*||
void compute_jacobian_residual(
        const MapState& map_state,
        double& mean_residual,
        double& max_residual)
{
    mean_residual = 0.0;
    max_residual = 0.0;
    int count = 0;

    // Get the mesh pair
    ISM_ASSERT_EQ(map_state.pairs_map_distortion.size(), 1);
    const int mesh_A_idx = map_state.pairs_map_distortion[0].first;
    const int mesh_B_idx = map_state.pairs_map_distortion[0].second;

    // For each face of T
    for (auto fh : map_state.mesh_T.faces())
    {
        // Get T-triangle vertices on sphere
        VH vh_a, vh_b, vh_c;
        handles(map_state.mesh_T, fh, vh_a, vh_b, vh_c);

        Vec3d a_sphere = map_state.embeddings_T[0][vh_a];
        Vec3d b_sphere = map_state.embeddings_T[0][vh_b];
        Vec3d c_sphere = map_state.embeddings_T[0][vh_c];

        // Lift to surfaces
        Vec3d a_lifted_A = lift_vertex_to_surface(a_sphere, map_state.meshes_input[mesh_A_idx],
            map_state.meshes_embeddings_input[mesh_A_idx], map_state.bsp_embeddings_input[mesh_A_idx]);
        Vec3d b_lifted_A = lift_vertex_to_surface(b_sphere, map_state.meshes_input[mesh_A_idx],
            map_state.meshes_embeddings_input[mesh_A_idx], map_state.bsp_embeddings_input[mesh_A_idx]);
        Vec3d c_lifted_A = lift_vertex_to_surface(c_sphere, map_state.meshes_input[mesh_A_idx],
            map_state.meshes_embeddings_input[mesh_A_idx], map_state.bsp_embeddings_input[mesh_A_idx]);

        Vec3d a_lifted_B = lift_vertex_to_surface(a_sphere, map_state.meshes_input[mesh_B_idx],
            map_state.meshes_embeddings_input[mesh_B_idx], map_state.bsp_embeddings_input[mesh_B_idx]);
        Vec3d b_lifted_B = lift_vertex_to_surface(b_sphere, map_state.meshes_input[mesh_B_idx],
            map_state.meshes_embeddings_input[mesh_B_idx], map_state.bsp_embeddings_input[mesh_B_idx]);
        Vec3d c_lifted_B = lift_vertex_to_surface(c_sphere, map_state.meshes_input[mesh_B_idx],
            map_state.meshes_embeddings_input[mesh_B_idx], map_state.bsp_embeddings_input[mesh_B_idx]);

        // Compute actual Jacobian
        Eigen::Vector2d a_local_A, b_local_A, c_local_A;
        Eigen::Vector2d a_local_B, b_local_B, c_local_B;
        to_local_coordinates(a_lifted_A, b_lifted_A, c_lifted_A, a_local_A, b_local_A, c_local_A);
        to_local_coordinates(a_lifted_B, b_lifted_B, c_lifted_B, a_local_B, b_local_B, c_local_B);

        Eigen::Matrix2d M_A;
        M_A << b_local_A - a_local_A, c_local_A - a_local_A;
        Eigen::Matrix2d M_B;
        M_B << b_local_B - a_local_B, c_local_B - a_local_B;

        if (M_A.determinant() <= 0 || M_B.determinant() <= 0)
            continue;

        Eigen::Matrix2d J_actual = M_B * M_A.inverse();

        // Look up prescribed Jacobian
        Eigen::Matrix2d J_star = lookup_prescribed_jacobian(
            a_sphere, b_sphere, c_sphere,
            a_lifted_A, b_lifted_A, c_lifted_A,
            a_lifted_B, b_lifted_B, c_lifted_B,
            0, map_state);

        // Compute residual
        double residual = (J_actual - J_star).norm();
        mean_residual += residual;
        max_residual = std::max(max_residual, residual);
        count++;
    }

    if (count > 0)
        mean_residual /= count;
}

void run()
{
    // Prepare output dir
    fs::path output_dir = OUTPUT_PATH / "prescribed_jacobian_test";
    fs::path screenshot_dir = output_dir / "screenshots";
    fs::create_directories(screenshot_dir);

    // Load meshes
    ISM_INFO("Loading L-shape meshes...");
    TriMesh mesh_A = read_mesh(DATA_PATH / "meshes/l_shape/newL.obj");
    TriMesh mesh_B = read_mesh(DATA_PATH / "meshes/l_shape/stretchedL.obj");

    ISM_INFO("Mesh A: " << mesh_A.n_vertices() << " vertices, " << mesh_A.n_faces() << " faces");
    ISM_INFO("Mesh B: " << mesh_B.n_vertices() << " vertices, " << mesh_B.n_faces() << " faces");

    ISM_ASSERT_EQ(mesh_A.n_vertices(), mesh_B.n_vertices());
    ISM_ASSERT_EQ(mesh_A.n_faces(), mesh_B.n_faces());

    // Normalize meshes
    TriMesh mesh_A_normalized = mesh_A;
    TriMesh mesh_B_normalized = mesh_B;
    normalize_mesh(mesh_A_normalized);
    normalize_mesh(mesh_B_normalized);

    // Extract the ground truth Jacobian field
    ISM_INFO("Extracting prescribed Jacobian field...");
    ExternalProperty<FH, PrescribedJacobian> jacobian_field =
        extract_jacobian_field(mesh_A_normalized, mesh_B_normalized);

    // Print some statistics about the Jacobian field
    double min_sigma = INF_DOUBLE, max_sigma = 0.0;
    for (auto fh : mesh_A_normalized.faces())
    {
        min_sigma = std::min(min_sigma, jacobian_field[fh].sigma.minCoeff());
        max_sigma = std::max(max_sigma, jacobian_field[fh].sigma.maxCoeff());
    }
    ISM_INFO("Jacobian field singular values: min=" << min_sigma << ", max=" << max_sigma);

    // Set up MapState
    // For this toy case, we need to create sphere embeddings
    // Since both meshes have identical connectivity, we can use the same embedding
    ISM_INFO("Setting up MapState with sphere embeddings...");

    // Use init_map for initialization
    MapState map_state;
    fs::path embedding_path_A = output_dir / "embedding_A.obj";
    fs::path embedding_path_B = output_dir / "embedding_B.obj";

    // Check if landmark files exist, otherwise create them
    fs::path landmarks_path_A = DATA_PATH / "meshes/l_shape/newL.pinned";
    fs::path landmarks_path_B = DATA_PATH / "meshes/l_shape/stretchedL.pinned";

    if (!fs::exists(landmarks_path_A) || !fs::exists(landmarks_path_B))
    {
        ISM_INFO("Creating landmark files with first 3 vertices...");

        // Create landmark files with first 3 vertices (matching indices since connectivity is identical)
        {
            std::ofstream ofs_A(landmarks_path_A);
            std::ofstream ofs_B(landmarks_path_B);
            for (int i = 0; i < 3; ++i)
            {
                ofs_A << i << "\n";
                ofs_B << i << "\n";
            }
        }
    }

    ISM_INFO("Initializing map with sphere embeddings...");
    init_map(map_state,
             { DATA_PATH / "meshes/l_shape/newL.obj", DATA_PATH / "meshes/l_shape/stretchedL.obj" },
             { landmarks_path_A, landmarks_path_B },
             { embedding_path_A, embedding_path_B },
             false);  // don't align meshes (keep original orientations)

    // Set up distortion pairs
    map_state.set_distortion_pairs(DistortionPairs::All);

    // Attach prescribed Jacobian field
    // The jacobian_field was computed on mesh_A_normalized, but map_state.meshes_input[0]
    // should be the same mesh (possibly with a different normalization).
    // For safety, let's recompute the field using the actual input meshes.
    ISM_INFO("Attaching prescribed Jacobian field to MapState...");
    ExternalProperty<FH, PrescribedJacobian> jacobian_field_state =
        extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]);
    map_state.prescribed_jacobians.push_back(jacobian_field_state);

    // Assign vertices to faces
    assign_vertices_to_T_faces(map_state);

    // Run landmark phase first to align landmarks
    ISM_INFO("Running landmark phase...");
    TinyAD::Timer timer_landmark("Landmark phase");
    landmark_phase(map_state);
    timer_landmark.stop();

    // First, run standard coarse phase to establish a good baseline
    // This fixes any inverted triangles from the landmark phase
    ISM_INFO("Running initial coarse phase (standard energy) to establish baseline...");
    TinyAD::Timer timer_coarse_init("Initial coarse phase");
    coarse_phase(map_state);
    timer_coarse_init.stop();

    // Now run with prescribed Jacobian to refine toward the target
    ISM_INFO("Running prescribed Jacobian refinement...");
    AdaptiveTriangulationsSettings settings = fine_phase_settings();
    settings.use_prescribed_jacobian = true;
    settings.max_iterations = 100;
    settings.w_approx = 1.0;
    settings.w_map = 1.0;
    settings.w_mesh = 0.5;

    TinyAD::Timer timer_coarse("Prescribed Jacobian phase");
    optimize_with_remeshing(map_state, settings);
    timer_coarse.stop();

    // Compute and print residual
    double mean_residual, max_residual;
    compute_jacobian_residual(map_state, mean_residual, max_residual);
    ISM_INFO("Jacobian residual after coarse phase: mean=" << mean_residual << ", max=" << max_residual);

    // Run fine phase with smaller approximation error
    ISM_INFO("Running fine phase with prescribed Jacobian energy...");
    settings = fine_phase_settings(0.0005);  // tighter approx error
    settings.use_prescribed_jacobian = true;
    settings.max_iterations = 50;

    TinyAD::Timer timer_fine("Fine phase (prescribed Jacobian)");
    optimize_with_remeshing(map_state, settings);
    timer_fine.stop();

    // Compute final residual
    compute_jacobian_residual(map_state, mean_residual, max_residual);
    ISM_INFO("Jacobian residual after fine phase: mean=" << mean_residual << ", max=" << max_residual);

    // Write output meshes
    ISM_INFO("Writing output meshes...");
    std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(map_state);
    for (int i = 0; i < (int)map_state.meshes_input.size(); ++i)
        write_mesh(lifted_Ts[i], output_dir / ("T_lifted_" + std::to_string(i) + ".obj"));

    // Visualization
    ISM_INFO("Generating visualization...");
    {
        auto style = default_style();
        auto g = gv::grid();

        // Left: Mesh A with lifted T overlay
        {
            auto v = gv::view();
            view_mesh(map_state.meshes_input[0], Color(0.8, 0.8, 0.8, 0.5));
            view_mesh(lifted_Ts[0], Color(1.0, 1.0, 1.0, 0.8));
            view_wireframe(lifted_Ts[0], MAGENTA, WidthScreen(0.5));
        }

        // Right: Mesh B with lifted T overlay
        {
            auto v = gv::view();
            view_mesh(map_state.meshes_input[1], Color(0.8, 0.8, 0.8, 0.5));
            view_mesh(lifted_Ts[1], Color(1.0, 1.0, 1.0, 0.8));
            view_wireframe(lifted_Ts[1], CYAN, WidthScreen(0.5));
        }
    }

    ISM_INFO("Total run time: " << timer_landmark.seconds() + timer_coarse_init.seconds() + timer_coarse.seconds() + timer_fine.seconds() << " seconds");
    ISM_INFO("Done!");
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
