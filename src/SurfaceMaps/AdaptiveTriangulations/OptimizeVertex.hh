/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */
#pragma once

#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTriangulationsSettings.hh>
#include <SurfaceMaps/Utils/Helpers.hh>

namespace SurfaceMaps
{

struct Triangle
{
    Triangle() = default;

    /// Init from current state (before an operation).
    /// Sets a, b, c in canonical order and reads vertex assignments
    /// from map state.
    Triangle(
            const FH& _fh,
            const TriMesh& _mesh,
            const ExternalProperty<VH, Vec3d>& _embedding_sphere,
            const ExternalProperty<FH, std::vector<SVH>>& _map_vhs,
            const AdaptiveTriangulationsSettings& _settings)
        : fh(_fh)
    {
        VH va, vb, vc;
        handles(_mesh, _fh, va, vb, vc);
        a_sphere = _embedding_sphere[va];
        b_sphere = _embedding_sphere[vb];
        c_sphere = _embedding_sphere[vc];

        if (_settings.w_approx > 0.0)
            assigned_vhs = _map_vhs[_fh];
    }

    /// Init candidate triangle (after an operation).
    /// fh doesn't have to exist yet.
    /// Set a, b, c in specific order. a should be the variable vertex.
    /// Leave vertex assignment empty.
    Triangle(
            const FH& _fh,
            const Vec3d& _a,
            const Vec3d& _b,
            const Vec3d& _c)
        : fh(_fh), a_sphere(_a), b_sphere(_b), c_sphere(_c) { }

    FH fh; // Current or future fh. Might not exist yet.

    Vec3d a_sphere; // In some functions this one gets replaced by variables
    Vec3d b_sphere;
    Vec3d c_sphere;

    std::vector<SVH> assigned_vhs; // A/B vertices assigned to this T triangle
};


/// Collect all A/B vertices assigned to a list of existing triangles.
std::vector<SVH> collect_assigned_vertices(
        const std::vector<Triangle>& _tris,
        const AdaptiveTriangulationsSettings& _settings);


/// Compute energy for list of triangles for n meshes and existing vertex assignments.
double eval_triangles(
        const std::vector<std::vector<Triangle>>& _tris,
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings);


/// Total objective over ALL of T, at the current state.
///
/// Built from eval_triangles over every face of T, so it is the same sum the optimizer
/// minimizes and the same one remeshing judges operations against -- barrier,
/// regularization, surface approximation, E_mesh and E_map. The landmark terms are not
/// included; they are zero once the landmarks are released, and the landmark phase is the
/// only place they are not.
///
/// Returns NaN before the first phase has run, when tels_input (the sizing field E_mesh
/// is measured against) does not exist yet.
///
/// It is an extensive quantity: a sum over |F(T)| faces, so it grows with refinement and
/// is only comparable across states with the same T. Divide by mesh_T.n_faces() to
/// compare across resolutions.
double total_energy(
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings);


/// Compute energy for list of triangles for n meshes.
/// Also compute new assignment of n meshes vertices to these triangles.
double eval_and_assign_triangles(
        std::vector<std::vector<Triangle> > &_tris, // assigns vertices
        std::vector<std::vector<SVH>> _total_vhs, //copy
        const MapState& _map_state,
        const AdaptiveTriangulationsSettings& _settings);

}
