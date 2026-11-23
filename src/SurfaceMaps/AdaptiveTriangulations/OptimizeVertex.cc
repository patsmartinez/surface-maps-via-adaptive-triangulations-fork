/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */

#include <SurfaceMaps/AdaptiveTriangulations/OptimizeVertex.hh>

#include <limits>

#include <SurfaceMaps/AdaptiveTriangulations/OptimizeMapEnergies.hh>
#include <SurfaceMaps/Misc/ConstantCurvatureGeometry.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTargetEdgeLength.hh>

namespace SurfaceMaps
{

std::vector<SVH> collect_assigned_vertices(
        const std::vector<Triangle>& _tris,
        const AdaptiveTriangulationsSettings& _settings)
{
    std::vector<SVH> vhs;
    if (_settings.w_approx <= 0.0)
        return vhs;

    for (const Triangle& tri : _tris)
        vhs.insert(vhs.end(), tri.assigned_vhs.begin(), tri.assigned_vhs.end());

    return vhs;
}

double total_energy(
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings)
{
    const int n_meshes = (int)_map_state.meshes_input.size();

    // The sizing field is built by init_target_edge_lengths, which runs at the start of
    // each optimize_with_remeshing call -- so before the first phase there is none, and
    // E_mesh is not defined yet. NaN rather than a guess, and rather than indexing an
    // empty tels_input, which is a segfault.
    // Every per-mesh property the energy reaches. tels_input is built by
    // init_target_edge_lengths and vertex_areas_input inside optimize_with_remeshing, so
    // before the first phase neither exists and E_mesh / E_approx are undefined.
    if ((int)_map_state.tels_input.size() != n_meshes
        || (int)_map_state.maps_Tf_inputvs.size() != n_meshes
        || (int)_map_state.vertex_areas_input.size() != n_meshes)
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<std::vector<Triangle>> tris(n_meshes);
    for (int i = 0; i < n_meshes; ++i)
    {
        tris[i].reserve(_map_state.mesh_T.n_faces());
        for (auto fh : _map_state.mesh_T.faces())
            tris[i].emplace_back(fh, _map_state.mesh_T, _map_state.embeddings_T[i],
                                 _map_state.maps_Tf_inputvs[i], _settings);
    }
    return eval_triangles(tris, _map_state, _settings);
}


double eval_triangles(
        const std::vector<std::vector<Triangle>>& _tris,
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings)
{
    ISM_ASSERT_EQ(_map_state.meshes_input.size(), _tris.size());
    int tri_num = _tris[0].size();
    for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
    {
        ISM_ASSERT_EQ(_tris[0].size(), _tris[i].size());
    }

    double E = 0.0;
    for (int tri_idx = 0; tri_idx < tri_num; ++tri_idx) // First singlemesh_energy for all triangles to check if infinity (hopefully speed up in some cases)
    {
        // Evaluate energy only affected by single mesh (i.e., barrier, regularization, surface approx)
        for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
        {
            E += eval_singlemesh_energy_triangle_T(_tris[i][tri_idx].a_sphere, _tris[i][tri_idx].b_sphere, _tris[i][tri_idx].c_sphere,
                                                   i, _map_state, _tris[i][tri_idx].assigned_vhs, _settings);
            if (E == INFINITY)
                return INFINITY;
        }
    }

    for (int tri_idx = 0; tri_idx < tri_num; ++tri_idx)
    {
        int mesh_tel_idx;
        double min_tel = INFINITY;
        // Mesh energy
        if (_settings.w_mesh > 0.0)
        {
            // Find minimal index for target edge length for mesh energy
            for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
            {
                double tel_i = ideal_triangle_edge_length(
                            _tris[i][tri_idx].a_sphere, _tris[i][tri_idx].b_sphere, _tris[i][tri_idx].c_sphere,
                            _map_state.meshes_input[i], _map_state.meshes_embeddings_input[i],
                            _map_state.tels_input[i], _map_state.bsp_embeddings_input[i]);
                if (tel_i < min_tel)
                {
                    mesh_tel_idx = i;
                    min_tel = tel_i;
                }
            }
            // Evaluate energy
            for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
            {
                // MUST match the ideal used by the Newton element in OptimizeMap.cc.
                // This path is what split/collapse/flip use to decide whether an
                // operation improves the energy; if the two disagreed, remeshing would be
                // judging a different energy than the one being minimized.
                E += eval_mesh_energy_triangle_T(_tris[i][tri_idx].a_sphere, _tris[i][tri_idx].b_sphere, _tris[i][tri_idx].c_sphere,
                                                 i, min_tel, _map_state, _settings);
                if (E == INFINITY)
                    return INFINITY;
            }
        }

        // Map energy, iterate over vector of pairs
        if (_settings.w_map > 0.0)
        {
            for (auto pair : _map_state.pairs_map_distortion)
            {
                const int mesh_A_idx = pair.first;
                const int mesh_B_idx = pair.second;
                E += eval_trianglepair_energy_triangle_T(_tris[mesh_A_idx][tri_idx].a_sphere, _tris[mesh_A_idx][tri_idx].b_sphere, _tris[mesh_A_idx][tri_idx].c_sphere,
                                                         _tris[mesh_B_idx][tri_idx].a_sphere, _tris[mesh_B_idx][tri_idx].b_sphere, _tris[mesh_B_idx][tri_idx].c_sphere,
                                                         mesh_A_idx, mesh_B_idx, _map_state, _settings);
                if (E == INFINITY)
                    return INFINITY;
            }
        }
    }
    return E;
}

double eval_and_assign_triangles(
        std::vector<std::vector<Triangle>>& _tris,
        std::vector<std::vector<SVH>> _total_vhs, //copy
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings)
{
    ISM_ASSERT_EQ(_map_state.meshes_input.size(), _tris.size());
    ISM_ASSERT_EQ(_map_state.meshes_input.size(), _total_vhs.size());
    for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
    {
        ISM_ASSERT_EQ(_tris[0].size(), _tris[i].size());
    }
    int tri_num = _tris[0].size();

    // Assign vertices to triangles
    if (_settings.w_approx > 0.0)
    {
        for (int tri_idx = 0; tri_idx < tri_num; tri_idx++)
        {
            for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
            {
                assign_vertices_to_triangle(_total_vhs[i], _tris[i][tri_idx].assigned_vhs, _tris[i][tri_idx].a_sphere, _tris[i][tri_idx].b_sphere, _tris[i][tri_idx].c_sphere, _map_state.embeddings_input[i]);
            }
        }
    }
    double E = eval_triangles(_tris, _map_state, _settings);

    // Make sure that all previously assigned vertices are also assigned for new configruation (if valid configuration)
    if (_settings.w_approx > 0.0 && E != INFINITY)
    {
        for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
        {
            ISM_ASSERT_EQ(_total_vhs[i].size(), 0);
        }
    }

    return E;
}


}
