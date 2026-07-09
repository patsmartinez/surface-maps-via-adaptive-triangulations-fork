/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */
#pragma once

#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>

#include <Eigen/SVD>

namespace SurfaceMaps
{

/// Extract the Jacobian field from two meshes with identical connectivity.
/// For each face of mesh_A, computes the SVD of the Jacobian mapping from A to B.
/// The SVD components (U, V, sigma) are stored in 3D ambient space.
///
/// @param mesh_A The source mesh
/// @param mesh_B The target mesh (must have identical connectivity to mesh_A)
/// @return ExternalProperty mapping each face to its PrescribedJacobian
ExternalProperty<FH, PrescribedJacobian> extract_jacobian_field(
        const TriMesh& mesh_A,
        const TriMesh& mesh_B);

/// Look up the prescribed Jacobian for a T-triangle.
/// Finds the face of mesh A that contains the T-triangle centroid on the sphere,
/// then projects the 3D SVD components into the T-triangle's local coordinate system.
///
/// @param _a_sphere, _b_sphere, _c_sphere T-triangle vertices on sphere
/// @param _a_lifted_A, _b_lifted_A, _c_lifted_A T-triangle vertices lifted to mesh A
/// @param _a_lifted_B, _b_lifted_B, _c_lifted_B T-triangle vertices lifted to mesh B
/// @param _pair_idx Index into pairs_map_distortion / prescribed_jacobians
/// @param _map_state The map state containing meshes and prescribed jacobians
/// @return The 2x2 Jacobian matrix in T-triangle's local coordinates
template <typename T>
Eigen::Matrix2<T> lookup_prescribed_jacobian(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const Vec3<T>& _a_lifted_B,
        const Vec3<T>& _b_lifted_B,
        const Vec3<T>& _c_lifted_B,
        const int _pair_idx,
        const MapState& _map_state);

/// Look up the prescribed pullback metric M* = V * Sigma^2 * V^T for a T-triangle.
/// Only the right singular vectors V (tangent to mesh A) and the singular values are needed;
/// the left singular vectors U are irrelevant to the metric form of the prescribed energy.
/// Finds the face of mesh A that contains the T-triangle centroid on the sphere,
/// then projects the 3D V vectors into the T-triangle's local coordinate system on A.
///
/// @param _a_sphere, _b_sphere, _c_sphere T-triangle vertices on sphere
/// @param _a_lifted_A, _b_lifted_A, _c_lifted_A T-triangle vertices lifted to mesh A
/// @param _pair_idx Index into pairs_map_distortion / prescribed_jacobians
/// @param _map_state The map state containing meshes and prescribed jacobians
/// @param _V_local Output: 2x2 rotation with V in T-triangle's local coordinates on A
/// @param _sigma Output: singular values
template <typename T>
void lookup_prescribed_metric(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const int _pair_idx,
        const MapState& _map_state,
        Eigen::Matrix2<T>& _V_local,
        Eigen::Vector2d& _sigma);

}
