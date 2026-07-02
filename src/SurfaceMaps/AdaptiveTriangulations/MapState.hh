/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */
#pragma once

#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/BSPTree.hh>

namespace SurfaceMaps
{

enum class DistortionPairs
{
    None,
    All,
    Sequence,
    Cycle,
    Star,
};

/// Stores the prescribed Jacobian field for a face of an input mesh.
/// The Jacobian describes the map from mesh A to mesh B.
/// We store the SVD components in 3D ambient space to allow proper
/// projection into arbitrary T-triangle local coordinate systems.
struct PrescribedJacobian
{
    // Right singular vectors (3D, tangent to surface A)
    // V columns define the principal directions on mesh A
    Eigen::Matrix<double, 3, 2> V;

    // Left singular vectors (3D, tangent to surface B)
    // U columns define the principal directions on mesh B
    Eigen::Matrix<double, 3, 2> U;

    // Singular values (sigma[0] >= sigma[1] > 0)
    Eigen::Vector2d sigma;
};

struct MapState
{
    MapState() = default;

    MapState(const MapState& _other)
        : meshes_input(_other.meshes_input),
          mesh_T(_other.mesh_T),
          embeddings_input(_other.embeddings_input),
          embeddings_T(_other.embeddings_T),
          landmarks_input(_other.landmarks_input),
          landmarks_T(_other.landmarks_T),
          tels_input(_other.tels_input),
          maps_Tf_inputvs(_other.maps_Tf_inputvs),
          vertex_areas_input(_other.vertex_areas_input),
          meshes_embeddings_input(_other.meshes_embeddings_input),
          prescribed_jacobians(_other.prescribed_jacobians)
    {
        compute_bsp_trees();
    }

    void compute_bsp_trees();

    // Set sphere embeddings and compute bsp trees
    void set_sphere_embeddings(
            const std::vector<ExternalProperty<VH, Vec3d>>& _embeddings);

    void set_distortion_pairs(
            const DistortionPairs& _pairs_mode);

    // Essential State Definition
    std::vector<TriMesh> meshes_input; // Fixed input geometries.
    TriMesh mesh_T; // Common triangulation T. Ignore vertex positions.

    std::vector<ExternalProperty<VH, Vec3d>> embeddings_input; // input embeddings on sphere.
    std::vector<ExternalProperty<VH, Vec3d>> embeddings_T; // T on spheres.

    std::vector<std::vector<VH>> landmarks_input; // Landmark vertexhandles for input meshes.
    std::vector<VH> landmarks_T;

    // Optional, Computable from Essential Info
    std::vector<ExternalProperty<VH, double>> tels_input; // target edge length for vertices of input meshes (e.g. for curvature based adaptive edge length)
    std::vector<ExternalProperty<FH, std::vector<SVH>>> maps_Tf_inputvs; // maps from each face of T to all vertices of corresponding input mesh in it
    std::vector<ExternalProperty<VH, double>> vertex_areas_input; // normalised vertex areas for vertices of input meshes
    std::vector<TriMesh> meshes_embeddings_input; // Redundant to meshes_input and embeddings_input. Needed for BSP trees. Can this be removed?
    std::vector<BSPTree> bsp_embeddings_input;

    std::vector<std::pair<int, int>> pairs_map_distortion;

    std::vector<ExternalProperty<VH, bool>> active_for_approx; // Specifies if a vertex should be considered for the surface approximation term, if not initialized for all meshes use all vertices by default

    // Prescribed Jacobian field per mesh pair.
    // Outer index: pair index (matching pairs_map_distortion).
    // Inner index: face index of the first mesh (A) in that pair.
    std::vector<ExternalProperty<FH, PrescribedJacobian>> prescribed_jacobians;
};

// Returns a BarycentricPoint on target mesh.
BarycentricPoint map_point(
        const MapState& _map_state,
        const int _idx_from,
        const int _idx_to,
        const BarycentricPoint& _bary_from,
        const TriMesh& _mesh_embedding_T_from,
        const BSPTree& _bsp_T_from);
}
