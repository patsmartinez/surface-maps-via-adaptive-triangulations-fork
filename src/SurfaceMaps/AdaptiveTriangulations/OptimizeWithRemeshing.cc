/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeWithRemeshing.hh>

#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTriangulationsSettings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Remeshing.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeMap.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTargetEdgeLength.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeMapEnergies.hh>

#include <TinyAD/Utils/Timer.hh>

namespace SurfaceMaps
{

double landmark_error(
        const MapState& _map_state)
{
    for(int i = 0; i < (int)_map_state.meshes_input.size(); i++)
    {
        ISM_ASSERT_EQ(_map_state.landmarks_input[i].size(), _map_state.landmarks_T.size());
    }

    double max_error = 0.0;
    for (int i = 0; i < (int)_map_state.landmarks_T.size(); ++i)
    {
        const VH v_T = _map_state.landmarks_T[i];
        for(int j = 0; j < (int)_map_state.meshes_input.size(); j++)
        {
            const VH v_j = _map_state.landmarks_input[j][i];
            const Vec3d p_T_j = _map_state.embeddings_T[j][v_T];
            const Vec3d p_j = lift_vertex_to_surface(p_T_j, _map_state.meshes_input[j], _map_state.meshes_embeddings_input[j], _map_state.bsp_embeddings_input[j]);
            const Vec3d p_j_target = _map_state.meshes_input[j].point(v_j);
            // Euclidean distance between current and target position in world space
            max_error = std::max(max_error, (p_j - p_j_target).norm());
        }
    }

    return max_error;
}



void init_target_edge_lengths(
        MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings)
{
    _map_state.tels_input.clear();

    // Input for A, compute factor for other meshes based on mesh areas
    std::vector<double> factor_A_to_i_area;
    for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
        factor_A_to_i_area.push_back(total_area(_map_state.meshes_input[i]) / total_area(_map_state.meshes_input[0]));

    for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
        _map_state.tels_input.push_back(target_edge_lengths(_map_state.meshes_input[i], _settings.approx_error * sqrt(factor_A_to_i_area[i]), _settings.min_edge_length * sqrt(factor_A_to_i_area[i]), _settings.max_edge_length * sqrt(factor_A_to_i_area[i])));

    ISM_ASSERT_EQ(_map_state.meshes_input.size(), _map_state.tels_input.size());

    // Map-driven refinement, folded into the SAME per-vertex field by min, so nothing
    // downstream changes: pick_target_edge_length, the remesher and E_mesh all just read
    // tels_input as before.
    //
    // Only mesh 0 gets it. The prescribed field describes the map A -> B and is indexed by
    // A's faces, so there is no counterpart on B -- and none is needed: the energy takes
    // fmin(tel_A, tel_B), so A's finer value wins wherever the band asks for it.
    if (_settings.map_driven_tel && !_map_state.prescribed_jacobians.empty())
    {
        ISM_ASSERT(_settings.mesh_tel_passive); // see settings.map_driven_tel

        int n_seam_verts = 0;
        ExternalProperty<VH, double> jump_field;
        ExternalProperty<VH, double> phi_field;
        // The far value is mesh 0's own curvature cap. factor_A_to_i_area[0] is exactly
        // 1, so this is settings.max_edge_length unscaled -- the two fields then agree
        // exactly outside the band and the min() is provably neutral there.
        const auto tel_map = map_driven_target_edge_lengths(
                    _map_state.meshes_input[0], _map_state.prescribed_jacobians[0],
                    _settings.map_tel_jump_thresh, _settings.map_tel_floor,
                    _settings.max_edge_length,
                    _settings.map_tel_diffusion_length,
                    n_seam_verts, &jump_field, &phi_field);

        _map_state.map_tel_phi = phi_field;

        int n_lowered = 0;
        for (auto vh : _map_state.meshes_input[0].vertices())
        {
            if (tel_map[vh] < _map_state.tels_input[0][vh])
            {
                _map_state.tels_input[0][vh] = tel_map[vh];
                ++n_lowered;
            }
        }

        // Area fraction of mesh A that the field drives genuinely fine, using the same
        // "< 1.5 x floor" rule the app's n_band_T uses on T. This, not n_lowered, is the
        // band: n_lowered counts a vertex nudged from 0.0300 to 0.0299 the same as one
        // taken to the floor.
        double band_area = 0.0;
        double total_a = 0.0;
        for (auto fh : _map_state.meshes_input[0].faces())
        {
            VH vh_a, vh_b, vh_c;
            handles(_map_state.meshes_input[0], fh, vh_a, vh_b, vh_c);
            const Vec3d& p_a = _map_state.meshes_input[0].point(vh_a);
            const double a = 0.5 * (_map_state.meshes_input[0].point(vh_b) - p_a)
                                   .cross(_map_state.meshes_input[0].point(vh_c) - p_a).norm();
            total_a += a;
            const double tel_f = (_map_state.tels_input[0][vh_a] + _map_state.tels_input[0][vh_b]
                                  + _map_state.tels_input[0][vh_c]) / 3.0;
            if (tel_f < 1.5 * _settings.map_tel_floor)
                band_area += a;
        }

        ISM_INFO("Map-driven tel: " << n_seam_verts << " seam vertices, "
                 << n_lowered << " of " << _map_state.meshes_input[0].n_vertices()
                 << " tels lowered (floor " << _settings.map_tel_floor << ", w "
                 << (_settings.map_tel_diffusion_length > 0.0
                     ? _settings.map_tel_diffusion_length
                     : map_tel_auto_diffusion_length(_map_state.meshes_input[0],
                                                     _settings.map_tel_floor))
                 << (_settings.map_tel_diffusion_length > 0.0 ? "" : " (auto)")
                 << "), band " << (total_a > 0.0 ? 100.0 * band_area / total_a : 0.0)
                 << "% of A's area");

        // Diagnostic dumps. Headless-safe (no GL), and the only way to see the field on
        // a shape where "does the band sit where I think" is not obvious by eye.
        if (!_settings.map_tel_debug_dir.empty())
        {
            const fs::path& dir = _settings.map_tel_debug_dir;
            fs::create_directories(dir);
            const TriMesh& mesh_A = _map_state.meshes_input[0];
            // Fixed range on the jump so runs stay comparable; auto on the other two.
            write_vertex_field_ply(mesh_A, jump_field, dir / "tel_01_jump.ply", 0.0,
                                   _settings.map_tel_jump_thresh * 4.0);
            write_vertex_field_ply(mesh_A, phi_field, dir / "tel_02_phi.ply", 0.0, 1.0);
            write_vertex_field_ply(mesh_A, _map_state.tels_input[0], dir / "tel_03_sizing.ply",
                                   _settings.map_tel_floor, _settings.max_edge_length);
        }

        if (n_seam_verts == 0)
            ISM_WARNING("Map-driven tel found no seam: the prescribed field varies less than map_tel_jump_thresh everywhere, so this is a no-op.");
    }
}


namespace
{

void optimize_with_remeshing_impl(
        MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings,
        const fs::path& _csv_path,
        std::function<void (const std::string&)> _callback_for_optim,
        std::function<void (const std::string&)> _callback_for_remesh)
{
    const int n_meshes = _map_state.meshes_input.size();

//    // Init CSV file
//    if (_cur_iter == 0 && !_csv_path.empty())
//    {
//        make_file_directory(_csv_path);
//        fs::remove(_csv_path);
//        if (!_settings.energies_in_csv)
//            append_to_csv(_csv_path,
//                          "iteration", "iter_type", "derivatives (seconds)", "solve (seconds)", "line search (seconds)", "iteration (seconds)", "objective",
//                          "splits (seconds)", "n_splits", "collapses (seconds)", "n_collapses", "flips (seconds)", "n_flips", "n_vertices");
//        else
//            ISM_ERROR_throw("Fixme");
//    }

    for (int i = 0; i < n_meshes; i++)
        ISM_ASSERT(_map_state.tels_input[i].size_okay(_map_state.meshes_input[i]));

    // Assign to each T triangle the vertices of input meshes that are inside
    if (_settings.w_approx > 0.0)
    {
        assign_vertices_to_T_faces(_map_state);
    }
    else
    {
        _map_state.maps_Tf_inputvs.clear();
        for (int i = 0; i < n_meshes; i++)
            _map_state.maps_Tf_inputvs.push_back(ExternalProperty<FH, std::vector<SVH>> (_map_state.mesh_T));
    }

    // Compute vertex areas for input meshes
    if (_map_state.vertex_areas_input.size() == 0) // Only compute if not already done previously
    {
        for (int i = 0; i < n_meshes; i++)
            _map_state.vertex_areas_input.push_back(rel_vertex_areas(_map_state.meshes_input[i]));

        // If active vertices specified, scale active vertex areas to still reach total
        if ((_map_state.active_for_approx.size() == _map_state.meshes_input.size()))
        {
            for (int i = 0; i < n_meshes; i++)
            {
                double total_area = 0.0;
                double active_area = 0.0;
                for (auto vh : _map_state.meshes_input[i].vertices())
                {
                    total_area += _map_state.vertex_areas_input[0][vh];
                    if (_map_state.active_for_approx[i][vh])
                        active_area += _map_state.vertex_areas_input[0][vh];
                }
                for (auto vh : _map_state.meshes_input[i].vertices())
                {
                    if (_map_state.active_for_approx[i][vh])
                        _map_state.vertex_areas_input[0][vh] *= total_area / active_area;
                }
            }
        }
    }

    _callback_for_optim("init");

    // Alternate optimize_map() and remesh_T()
    double time_remeshing = 0.0;
    double time_optimize = 0.0;
    OptimizeMapStatus continuous_opt_status = normal;
    for (int iter = 0; iter < _settings.max_iterations; ++iter)
    {
        ISM_INFO("Starting iteration: " << iter);

        // Remeshing step
        bool remesh_changed_map_state = false;
        TinyAD::Timer timer_remeshing("remeshing");
        remesh_changed_map_state = remesh_T(_map_state, _settings, _csv_path, _callback_for_remesh);
        timer_remeshing.stop();
        time_remeshing += timer_remeshing.seconds();

        // Sanity checks
        ISM_ASSERT(_settings.w_approx <= 0.0 || sanity_check_all_vertices_assigned(_map_state));
        ISM_ASSERT(_settings.w_approx <= 0.0 || sanity_check_assignments(_map_state));

//        // Print stats
//        {
//            ISM_DEBUG_OUT("Vertices after remeshing: " << _map_state.mesh_T.n_vertices());
//            ISM_DEBUG_OUT("Faces after remeshing: " << _map_state.mesh_T.n_faces());
//            double min_el = INF_DOUBLE;
//            double max_el = 0.0;
//            double avg_el = 0.0;
//            std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(_map_state);
//            for (int i = 0; i < n_meshes; i++)
//            {
//                for (auto e : _map_state.mesh_T.edges())
//                {
//                    const double el = lifted_Ts[i].calc_edge_length(e);
//                    min_el = std::min(min_el, el);
//                    max_el = std::max(min_el, el);
//                    avg_el += el;
//                }
//            }
//            avg_el /= n_meshes * _map_state.mesh_T.n_edges();
//            ISM_DEBUG_VAR(min_el);
//            ISM_DEBUG_VAR(max_el);
//            ISM_DEBUG_VAR(avg_el);
//        }

        // Check convergence after remeshing!
        if (!remesh_changed_map_state && continuous_opt_status == under_threshold)
        {
            ISM_HIGHLIGHT("Continous optimization and remeshing converged in iteration " << iter) ;
            break;
        }

        // No more progress?
        if (!remesh_changed_map_state && continuous_opt_status == line_search_converge)
        {
            ISM_HIGHLIGHT("Line search couldn't find improvement and remeshing converged in iteration " << iter);
            break;
        }

        // Landmarks satisfied?
        const double landmark_err = landmark_error(_map_state);
        ISM_DEBUG_VAR(landmark_err);
        if (!remesh_changed_map_state && landmark_error(_map_state) < _settings.landmark_threshold)
        {
            ISM_HIGHLIGHT("Landmark error under " << _settings.landmark_threshold << " in iteration " << iter);
            break;
        }

        // Continuous optimization
        TinyAD::Timer timer_optimize("optimize");
        continuous_opt_status = optimize_map(_map_state, _settings.optimize_per_remeshing_iters, _callback_for_optim, _csv_path, _settings);
        timer_optimize.stop();
        time_optimize += timer_optimize.seconds();

        // Sanity checks
        ISM_ASSERT(_settings.w_approx <= 0.0 || sanity_check_all_vertices_assigned(_map_state));
        ISM_ASSERT(_settings.w_approx <= 0.0 || sanity_check_assignments(_map_state));
    }
}

}

void optimize_with_remeshing(
        MapState& _map_state,
        const std::vector<ExternalProperty<VH, double>>& _mesh_tels,
        const AdaptiveTriangulationsSettings& _settings,
        const fs::path& _csv_path,
        std::function<void (const std::string&)> _callback_for_optim,
        std::function<void (const std::string&)> _callback_for_remesh)
{
    _map_state.tels_input = _mesh_tels;
    optimize_with_remeshing_impl(_map_state, _settings, _csv_path, _callback_for_optim, _callback_for_remesh);
}

void optimize_with_remeshing(
        MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings,
        const fs::path& _csv_path,
        std::function<void (const std::string&)> _callback_for_optim,
        std::function<void (const std::string&)> _callback_for_remesh)
{
    init_target_edge_lengths(_map_state, _settings);
    optimize_with_remeshing_impl(_map_state, _settings, _csv_path, _callback_for_optim, _callback_for_remesh);
}

}
