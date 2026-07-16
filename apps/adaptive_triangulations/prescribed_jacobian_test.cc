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
#include <glow-extras/viewer/canvas.hh>
#include <glow-extras/viewer/experimental.hh>
#include <imgui/imgui.h>
#include <fstream>

namespace SurfaceMaps
{

/// Validate the extracted Jacobian field with zero optimization involved:
/// reconstructing B's triangles from A's via J* must succeed to machine precision,
/// since both meshes share connectivity and J* is exact per face.
void validate_jacobian_field(
        const TriMesh& mesh_A,
        const TriMesh& mesh_B,
        const ExternalProperty<FH, PrescribedJacobian>& field)
{
    double max_error = 0.0;

    for (auto fh : mesh_A.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(mesh_A, fh, vh_a, vh_b, vh_c);

        Vec3d a_A = mesh_A.point(vh_a);
        Vec3d b_A = mesh_A.point(vh_b);
        Vec3d c_A = mesh_A.point(vh_c);
        Vec3d a_B = mesh_B.point(vh_a);
        Vec3d b_B = mesh_B.point(vh_b);
        Vec3d c_B = mesh_B.point(vh_c);

        // Local frames and edge matrices, exactly as in extract_jacobian_field
        Eigen::Vector2d a_local_A, b_local_A, c_local_A;
        Eigen::Vector2d a_local_B, b_local_B, c_local_B;
        to_local_coordinates(a_A, b_A, c_A, a_local_A, b_local_A, c_local_A);
        to_local_coordinates(a_B, b_B, c_B, a_local_B, b_local_B, c_local_B);

        Eigen::Matrix2d M_A;
        M_A << b_local_A - a_local_A, c_local_A - a_local_A;
        Eigen::Matrix2d M_B;
        M_B << b_local_B - a_local_B, c_local_B - a_local_B;

        // Project the stored 3D singular vectors back into the face frames
        Vec3d normal_A = ((b_A - a_A).cross(c_A - a_A)).normalized();
        Vec3d basis0_A = (b_A - a_A).normalized();
        Vec3d basis1_A = normal_A.cross(basis0_A);

        Vec3d normal_B = ((b_B - a_B).cross(c_B - a_B)).normalized();
        Vec3d basis0_B = (b_B - a_B).normalized();
        Vec3d basis1_B = normal_B.cross(basis0_B);

        const PrescribedJacobian& pj = field[fh];
        Eigen::Matrix2d V_2d, U_2d;
        for (int i = 0; i < 2; ++i)
        {
            V_2d(0, i) = basis0_A.dot(pj.V.col(i));
            V_2d(1, i) = basis1_A.dot(pj.V.col(i));
            U_2d(0, i) = basis0_B.dot(pj.U.col(i));
            U_2d(1, i) = basis1_B.dot(pj.U.col(i));
        }

        // Reconstruct J* and check that it maps A's edges onto B's edges
        Eigen::Matrix2d J_star = U_2d * pj.sigma.asDiagonal() * V_2d.transpose();
        max_error = std::max(max_error, (J_star * M_A - M_B).norm());
    }

    ISM_INFO("Jacobian field validation: max reconstruction error = " << max_error);
    ISM_ASSERT_L(max_error, 1e-8);
}

/// Compute the actual and prescribed Jacobians of a single T-face.
/// Returns false if the face is degenerate on either surface or J* is near-singular.
bool compute_T_face_jacobians(
        const MapState& map_state,
        const FH fh,
        Eigen::Matrix2d& J_actual,
        Eigen::Matrix2d& J_star)
{
    ISM_ASSERT_EQ(map_state.pairs_map_distortion.size(), 1);
    const int mesh_A_idx = map_state.pairs_map_distortion[0].first;
    const int mesh_B_idx = map_state.pairs_map_distortion[0].second;

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
        return false;

    J_actual = M_B * M_A.inverse();

    // Look up prescribed Jacobian
    J_star = lookup_prescribed_jacobian(
        a_sphere, b_sphere, c_sphere,
        a_lifted_A, b_lifted_A, c_lifted_A,
        a_lifted_B, b_lifted_B, c_lifted_B,
        0, map_state);

    if (std::abs(J_star.determinant()) <= 1e-10)
        return false;

    return true;
}

/// Compute the rotation-invariant residual of the map against the prescribed field:
/// the singular values of J * J*^{-1} per T-face, which are all 1 exactly when the
/// pullback metric matches the prescription. (A Frobenius norm ||J - J*|| is misleading
/// here: the energy is invariant under J -> R*J*, so it can stay large at a perfect
/// metric match.) Reports mean/max |sigma - 1| and the fraction of faces over 0.1,
/// and optionally the per-face max deviation for heatmap visualization.
void compute_metric_residual(
        const MapState& map_state,
        double& mean_deviation,
        double& max_deviation,
        double& frac_over_thresh,
        ExternalProperty<FH, double>* deviation_per_face = nullptr)
{
    mean_deviation = 0.0;
    max_deviation = 0.0;
    frac_over_thresh = 0.0;
    int count = 0;

    if (deviation_per_face)
        *deviation_per_face = ExternalProperty<FH, double>(map_state.mesh_T, 0.0);

    // For each face of T
    for (auto fh : map_state.mesh_T.faces())
    {
        Eigen::Matrix2d J_actual, J_star;
        if (!compute_T_face_jacobians(map_state, fh, J_actual, J_star))
            continue;

        // Singular values of the residual J * J*^{-1}: both 1 iff the metric matches
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(J_actual * J_star.inverse());
        const double dev = std::max(std::abs(svd.singularValues()[0] - 1.0),
                                    std::abs(svd.singularValues()[1] - 1.0));

        mean_deviation += std::abs(svd.singularValues()[0] - 1.0) + std::abs(svd.singularValues()[1] - 1.0);
        max_deviation = std::max(max_deviation, dev);
        if (dev > 0.1)
            frac_over_thresh += 1.0;
        if (deviation_per_face)
            (*deviation_per_face)[fh] = dev;
        count++;
    }

    if (count > 0)
    {
        mean_deviation /= 2.0 * count;
        frac_over_thresh /= count;
    }
}

/// Pick the face of a mesh closest to the mouse cursor's world position.
/// Follows the pattern of pick_vertex in Viewer/Picking.cc.
/// Returns an invalid handle (and infinite distance) if the mouse is not over geometry.
FH pick_face(
        const TriMesh& _mesh,
        double& _dist_sqr)
{
    using namespace gv::experimental;
    _dist_sqr = INF_DOUBLE;

    // Don't pick if UI captures the mouse click
    if (ImGui::GetIO().WantCaptureMouse)
        return FH(-1);

    // Get mouse position
    auto p_world = interactive_get_position(interactive_get_mouse_position());

    if (!p_world.has_value() || _mesh.n_faces() == 0)
        return FH(-1);

    const tg::pos3 p = p_world.value();

    // Return face with the closest point-to-triangle distance
    FH fh_best(-1);
    for (auto fh : _mesh.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_mesh, fh, vh_a, vh_b, vh_c);
        const tg::triangle3 tri(tg::pos3(_mesh.point(vh_a)), tg::pos3(_mesh.point(vh_b)), tg::pos3(_mesh.point(vh_c)));
        const double d = tg::distance_sqr(p, tg::project(p, tri));
        if (d < _dist_sqr)
        {
            _dist_sqr = d;
            fh_best = fh;
        }
    }
    return fh_best;
}

/// Draw a face highlight slightly offset along the face normal (avoids z-fighting).
void highlight_face(
        const TriMesh& _mesh,
        const int _face_idx,
        gv::canvas_t& _c)
{
    if (_face_idx < 0 || _face_idx >= (int)_mesh.n_faces())
        return;

    VH vh_a, vh_b, vh_c;
    handles(_mesh, FH(_face_idx), vh_a, vh_b, vh_c);

    const Vec3d a = _mesh.point(vh_a);
    const Vec3d b = _mesh.point(vh_b);
    const Vec3d c = _mesh.point(vh_c);
    const Vec3d offset = 1e-3 * ((b - a).cross(c - a)).normalized();

    _c.add_face(tg::pos3(a + offset), tg::pos3(b + offset), tg::pos3(c + offset), tg::color3(YELLOW));
    _c.set_line_width_world(0.002);
    _c.add_line(tg::pos3(a + offset), tg::pos3(b + offset), tg::color3(YELLOW));
    _c.add_line(tg::pos3(b + offset), tg::pos3(c + offset), tg::color3(YELLOW));
    _c.add_line(tg::pos3(c + offset), tg::pos3(a + offset), tg::color3(YELLOW));
}

void run()
{
    // Tunable weights for the prescribed phases.
    // w_mesh trades triangle quality against the prescription: high values fight the
    // prescribed anisotropy (E_mesh wants equilateral triangles on both surfaces),
    // but 0 lets triangles drift freely wherever the prescription is isotropic
    // (in-plane sliding/rotation on flat regions costs no map energy).
    const double w_mesh_stage1 = 0.25;
    const double w_mesh_stage2 = 0.1;
    const int max_iterations_stage1 = 100;
    const int max_iterations_stage2 = 50;

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

    // Sanity-check the field before any optimization: reconstructing B from A + J*
    // must succeed to machine precision.
    validate_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1], jacobian_field_state);

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

    // Baseline residual: the standard energy targets isometry, not the prescription,
    // so this should be clearly nonzero.
    double mean_dev, max_dev, frac_bad;
    compute_metric_residual(map_state, mean_dev, max_dev, frac_bad);
    ISM_INFO("Metric residual |sigma(J J*^-1) - 1| after coarse phase (baseline): "
             << "mean=" << mean_dev << ", max=" << max_dev << ", frac>0.1: " << frac_bad);

    // Stage 1: continuous phase only, no remeshing.
    ISM_INFO("Running prescribed Jacobian phase (metric form, remeshing disabled)...");
    AdaptiveTriangulationsSettings settings = fine_phase_settings();
    settings.use_prescribed_jacobian = true;
    settings.allow_splits = false;
    settings.allow_collapses = false;
    settings.allow_flips = false;
    settings.w_mesh = w_mesh_stage1;
    settings.max_iterations = max_iterations_stage1;
    settings.w_approx = 1.0;
    settings.w_map = 1.0;

    TinyAD::Timer timer_coarse("Prescribed Jacobian phase (continuous only)");
    optimize_with_remeshing(map_state, settings);
    timer_coarse.stop();

    compute_metric_residual(map_state, mean_dev, max_dev, frac_bad);
    ISM_INFO("Metric residual |sigma(J J*^-1) - 1| after continuous phase: "
             << "mean=" << mean_dev << ", max=" << max_dev << ", frac>0.1: " << frac_bad);

    // Stage 2: enable remeshing with a lowered mesh weight, since the equilateral
    // target of E_mesh fights the prescribed anisotropy.
    ISM_INFO("Running fine phase with prescribed Jacobian energy (remeshing enabled)...");
    settings = fine_phase_settings(0.0005);  // tighter approx error
    settings.use_prescribed_jacobian = true;
    settings.w_mesh = w_mesh_stage2;
    settings.max_iterations = max_iterations_stage2;

    TinyAD::Timer timer_fine("Fine phase (prescribed Jacobian)");
    optimize_with_remeshing(map_state, settings);
    timer_fine.stop();

    // Final residual, keeping the per-face deviation for the heatmap
    ExternalProperty<FH, double> deviation_per_face;
    compute_metric_residual(map_state, mean_dev, max_dev, frac_bad, &deviation_per_face);
    ISM_INFO("Metric residual |sigma(J J*^-1) - 1| after fine phase: "
             << "mean=" << mean_dev << ", max=" << max_dev << ", frac>0.1: " << frac_bad);

    // Write output meshes
    ISM_INFO("Writing output meshes...");
    std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(map_state);
    for (int i = 0; i < (int)map_state.meshes_input.size(); ++i)
        write_mesh(lifted_Ts[i], output_dir / ("T_lifted_" + std::to_string(i) + ".obj"));

    // Visualization
    // Both lifted meshes share mesh_T's connectivity, so a face selected on one pane
    // is the corresponding face on the other. Middle-click a triangle to inspect it.
    ISM_INFO("Generating visualization...");
    ISM_INFO("Middle-click or double-click a triangle (in any pane) to highlight it on both meshes and show its singular values.");
    {
        auto style = default_style();

        int selected_face = -1;

        gv::interactive([&] (auto)
        {
            // Middle-click or double-click: select the T-face under the mouse. The picked
            // world position lies on whichever lifted mesh is hovered, so take the closer
            // of the two.
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
            {
                double dist_0, dist_1;
                const FH fh_0 = pick_face(lifted_Ts[0], dist_0);
                const FH fh_1 = pick_face(lifted_Ts[1], dist_1);
                const FH fh_pick = (dist_0 <= dist_1) ? fh_0 : fh_1;
                if (fh_pick.is_valid())
                    selected_face = fh_pick.idx();
            }

            // Info window for the selected triangle
            if (selected_face >= 0)
            {
                const FH fh(selected_face);
                ImGui::Begin("Selected triangle");
                ImGui::Text("T-face %d", selected_face);

                Eigen::Matrix2d J_actual, J_star;
                if (compute_T_face_jacobians(map_state, fh, J_actual, J_star))
                {
                    const Eigen::Vector2d s_actual = Eigen::JacobiSVD<Eigen::Matrix2d>(J_actual).singularValues();
                    const Eigen::Vector2d s_target = Eigen::JacobiSVD<Eigen::Matrix2d>(J_star).singularValues();
                    const Eigen::Vector2d s_residual = Eigen::JacobiSVD<Eigen::Matrix2d>(J_actual * J_star.inverse()).singularValues();

                    ImGui::Text("sigma(J) actual:      %.4f  %.4f", s_actual[0], s_actual[1]);
                    ImGui::Text("sigma(J*) prescribed: %.4f  %.4f", s_target[0], s_target[1]);
                    ImGui::Separator();
                    ImGui::Text("sigma(J J*^-1):       %.4f  %.4f", s_residual[0], s_residual[1]);
                    ImGui::Text("deviation from 1:     %.4f",
                                std::max(std::abs(s_residual[0] - 1.0), std::abs(s_residual[1] - 1.0)));
                }
                else
                {
                    ImGui::Text("Degenerate triangle or prescription.");
                }
                ImGui::End();
            }

            auto g = gv::grid();

            // Left: Mesh A with lifted T overlay
            {
                auto v = gv::view();
                auto c = gv::canvas();
                highlight_face(lifted_Ts[0], selected_face, c);
                view_mesh(map_state.meshes_input[0], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_Ts[0], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_Ts[0], MAGENTA, WidthScreen(0.5));
            }

            // Right: Mesh B with lifted T overlay
            {
                auto v = gv::view();
                auto c = gv::canvas();
                highlight_face(lifted_Ts[1], selected_face, c);
                view_mesh(map_state.meshes_input[1], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_Ts[1], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_Ts[1], TEAL, WidthScreen(0.5));
            }

            // Third: metric residual heatmap on B.
            // White = pullback metric matches the prescription; magenta = deviation >= 0.5.
            // Color off the stretched arm means real smearing; color only along the
            // transition band means T-triangles straddling the prescription jump.
            {
                auto v = gv::view();
                auto colors = linear_colors(deviation_per_face, 0.0, 0.5, WHITE, MAGENTA);
                gv::view(make_renderable(lifted_Ts[1], colors));
            }
        });
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
