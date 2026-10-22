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
#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTriangulationsSettings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>

namespace SurfaceMaps
{

/// This prevents triangles from degenerating or inverting or spanning a full hemisphere, and edges from approaching length pi
template <typename T>
T barrier_energy(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere);

/// Compute surface approx energy for all A or B vertices in a T triangle
template <typename T>
T surface_approx_barrier(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const Vec3<T>& _a_lifted,
        const Vec3<T>& _b_lifted,
        const Vec3<T>& _c_lifted,
        const std::vector<SVH>& _vhs_X,
        const TriMesh& _mesh_X,
        const ExternalProperty<VH, Vec3d>& _embedding_X,
        const ExternalProperty<VH, double>& _vertex_areas_X,
        const double& _bound);

/// Evaluate energy that only concern a single mesh for a triangle of mesh T (i.e., barrier, regulaization, surface approx)
template <typename T>
T eval_singlemesh_energy_triangle_T(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const int& _mesh_idx,
        const MapState& _map_state,
        const std::vector<SVH>& _vhs_A, // A vertices in triangle of T
        const AdaptiveTriangulationsSettings& _settings);

/// Standard symmetric Dirichlet map energy
template <typename T>
T map_energy(
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const Vec3<T>& _a_lifted_B,
        const Vec3<T>& _b_lifted_B,
        const Vec3<T>& _c_lifted_B);

/// Prescribed Jacobian map energy - penalizes deviation from target Jacobian J*
/// Computes symmetric Dirichlet energy on the residual Jacobian: J_residual = J * J*^{-1}
/// When J = J*, J_residual = I and the energy is minimal.
template <typename T>
T map_energy_prescribed(
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const Vec3<T>& _a_lifted_B,
        const Vec3<T>& _b_lifted_B,
        const Vec3<T>& _c_lifted_B,
        const Eigen::Matrix2<T>& _J_star);

/// Metric form of the prescribed Jacobian map energy.
/// Prescribes only the pullback metric M* = V * Sigma^2 * V^T (i.e., J^T J), not the full Jacobian:
/// E = area_B * tr(J M*^{-1} J^T) + area_A * tr(J^{-1} J^{-T} M*)
/// computed in the factored form area_B * ||J V Sigma^{-1}||^2 + area_A * ||Sigma V^T J^{-1}||^2.
/// Identical energy landscape to map_energy_prescribed (Frobenius norms are invariant to U),
/// but the left singular vectors U are never needed.
template <typename T>
T map_energy_prescribed_metric(
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const Vec3<T>& _a_lifted_B,
        const Vec3<T>& _b_lifted_B,
        const Vec3<T>& _c_lifted_B,
        const Eigen::Matrix2<T>& _V_local,
        const Eigen::Vector2d& _sigma);

/// Evaluate energy that concerns pair of triangles on two meshes for a triangle of mesh T (i.e., map energy)
template <typename T>
T eval_trianglepair_energy_triangle_T(
        const Vec3<T>& _a_sphere_A,
        const Vec3<T>& _b_sphere_A,
        const Vec3<T>& _c_sphere_A,
        const Vec3<T>& _a_sphere_B,
        const Vec3<T>& _b_sphere_B,
        const Vec3<T>& _c_sphere_B,
        const int& _mesh_A_idx,
        const int& _mesh_B_idx,
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings);

/// Index of the prescribed pair whose B side is _mesh_idx, or -1 if this mesh is not the
/// B side of one (or the prescription-aware mesh ideal is off). Callers need it before
/// prescribed_ideal_shape, to know which mesh's sphere positions to fetch.
int prescribed_pair_for_B_mesh(
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings,
        const int _mesh_idx);

/// The 2x2 shape matrix E_mesh should apply to the ideal triangle on the B side of
/// _pair_idx. SPD with det == 1: it carries the prescribed ANISOTROPY and leaves the AREA
/// to the sizing field. Identity if the lookup degenerates.
///
/// Takes the A-side sphere positions of the T triangle, because the prescribed field is
/// indexed by faces of mesh A and the returned matrix lives in that triangle's local
/// frame on A.
Eigen::Matrix2d prescribed_ideal_shape(
        const Vec3d& _a_sphere_A,
        const Vec3d& _b_sphere_A,
        const Vec3d& _c_sphere_A,
        const int _pair_idx,
        const MapState& _map_state);

/// Evaluate mesh energy for triangle of T for MapState.
/// _ideal_shape deforms the ideal triangle; identity gives the original equilateral.
template <typename T>
T eval_mesh_energy_triangle_T(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const int& _mesh_idx,
        const T& _tel,
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings,
        const Eigen::Matrix2d& _ideal_shape = Eigen::Matrix2d::Identity());

/// Evaluate energy for a landmark vertex of mesh T with two embeddings
template <typename T>
T eval_landmark_T_energy(
        const Vec3<T>& _p_sphere,
        const VH& _vh_input,
        const int& _idx_input,
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings);

}
