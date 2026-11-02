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
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/Utils/BSPTree.hh>

namespace SurfaceMaps
{

/// compute adaptive target edge length for a mesh
ExternalProperty<VH, double> target_edge_lengths(
        const TriMesh& _mesh,
        const double _approx_error,
        const double _min_edge_length,
        const double _max_edge_length);

/// MAP-DRIVEN target edge length: refine where the PRESCRIBED metric changes, not only
/// where the surface curves. Prototype; combined with the curvature field by min.
///
/// Why this exists. The curvature field says nothing about the map, so on a flat region it
/// allows arbitrarily long edges no matter what the prescription does there. That is what
/// the max_edge_length cap was papering over: capping it globally at 0.03 took the U from
/// 0.90x to 6.15x, at 2.3x the faces EVERYWHERE. Refining only where the prescription
/// actually varies is the targeted version of the same lever -- measured cost on the L is
/// ~1.55x faces for a floor of 0.015, against 2.3x for the global cap.
///
/// What it measures. The prescribed field is piecewise CONSTANT per face of A, so its
/// variation is not a gradient but a jump living on edges. For each interior edge the
/// discrepancy is taken between the two faces' SINGULAR VALUES only,
///     d = sqrt( log(s0_f0/s0_f1)^2 + log(s1_f0/s1_f1)^2 ),
/// which is frame-free. Comparing the full ambient V diag(sigma^2) V^T instead would
/// charge every CREASE as a prescription jump -- the L's creases are 45-70 degrees -- and
/// that is geometry, not prescription. The price is that sigma alone cannot see a field
/// whose principal DIRECTIONS rotate at constant stretch; on all three test shapes the
/// seam does move sigma, so it is detected, but a direction-aware version needs V
/// parallel-transported across the edge.
///
/// Three choices that are not arbitrary:
///  - MAX over a vertex's incident edges, never the average. tel is a "no longer than"
///    requirement, so the binding constraint at a vertex is its worst neighbour;
///    averaging would dilute a big jump against a small one and under-refine. Same reason
///    mesh_energy takes fmin(tel_A, tel_B) rather than their mean.
///  - DILATION by _min_rings. Not cosmetic: tel is read by ONE point sample at each T
///    triangle's sphere centroid (ideal_triangle_edge_length), so a band narrower than
///    the elements it is trying to refine is invisible -- most centroids land outside it
///    and never see the fine value. Measured, the L's input edges are 0.0629 and the U's
///    0.0149, so on the U a single ring is about one element wide and marginal; two
///    rings is the safe default.
///  - GRADATION, i.e. tel = floor + g * dist(seam) by Dijkstra, rather than a fixed
///    number of rings. g is a size-per-length and so is independent of input resolution,
///    where "one ring" is not (0.0629 vs 0.0149 is a 4x different gradation for the same
///    rule). It is also one-sided: it only ever RAISES tel away from the seam and can
///    never lift the floor inside the band, which is exactly why this and not Laplacian
///    smoothing. Measured, the transition is nearly free -- g from 0.25 to infinity spans
///    only 3-12% of face count, which is all set by _floor instead.
///
/// Returns INF_DOUBLE on every vertex the seam does not reach, so that
/// min(tel_curvature, tel_map) leaves the curvature field untouched there. An empty seam
/// therefore makes this a no-op.
ExternalProperty<VH, double> map_driven_target_edge_lengths(
        const TriMesh& _mesh_A,
        const ExternalProperty<FH, PrescribedJacobian>& _prescribed,
        const double _jump_threshold,
        const double _floor,
        const double _gradation,
        const int _min_rings,
        int& _out_n_seam_verts);

/// Pick target edge length for single vertex of T (for one _mesh_input)
template <typename T>
T pick_target_edge_length(
        const Vec3<T>& _p_sphere,
        const TriMesh& _mesh_input, // mesh_A or mesh_B
        const TriMesh& _mesh_input_embedding,
        const ExternalProperty<VH, double>& _mesh_input_tels,
        const BSPTree& _bsp_input)
{
    SFH f;
    T alpha, beta, gamma;
    bsp_tree_barys_face(_p_sphere, _mesh_input_embedding, _bsp_input, alpha, beta, gamma, f);

    // Interpolate p/w lin scalar field
    SVH vh_a, vh_b, vh_c;
    handles(_mesh_input, f, vh_a, vh_b, vh_c);
    T interpol_tel = _mesh_input_tels[vh_a] * alpha + _mesh_input_tels[vh_b] * beta + _mesh_input_tels[vh_c] * gamma;

    return interpol_tel;
}

/// Target edge length for single face from its three vertex target edge lengths
template <typename T>
T triangle_tel_from_vertex_tels(
        const T& _tel_v1,
        const T& _tel_v2,
        const T& _tel_v3,
        const bool _pick_minimum)
{
    if (_pick_minimum)
        return fmin(_tel_v1, fmin(_tel_v2, _tel_v3));
    else
        return (_tel_v1 + _tel_v2 + _tel_v3) / 3.0; // avg of vertices tel
}

/// Target edge length for triangle of T (on one _mesh_input)
template <typename T>
T target_edge_length(
        const Vec3<T>& _v0_sphere,
        const Vec3<T>& _v1_sphere,
        const TriMesh& _mesh_input, // mesh_A or mesh_B
        const TriMesh& _mesh_input_embedding,
        const ExternalProperty<VH, double>& _mesh_input_tels,
        const BSPTree& _bsp_input)
{
    Vec3<T> p = (_v0_sphere + _v1_sphere ) * 0.5; // Skip normalization because only used for barycentric coordinates
    return pick_target_edge_length(p, _mesh_input, _mesh_input_embedding, _mesh_input_tels, _bsp_input);
}

/// Target edge length for triangle of T (on one _mesh_input)
template <typename T>
T ideal_triangle_edge_length(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const TriMesh& _mesh_input, // mesh_A or mesh_B
        const TriMesh& _mesh_input_embedding,
        const ExternalProperty<VH, double>& _mesh_input_tels,
        const BSPTree& _bsp_input)
{
    Vec3<T> middle_p = (_a_sphere + _b_sphere + _c_sphere) * (1.0/3.0); // Skip normalization because only used for barycentric coordinates
    return pick_target_edge_length(middle_p, _mesh_input, _mesh_input_embedding, _mesh_input_tels, _bsp_input);
}

}
