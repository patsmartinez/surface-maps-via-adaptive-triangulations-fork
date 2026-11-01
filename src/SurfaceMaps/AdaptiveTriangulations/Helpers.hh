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

namespace SurfaceMaps
{

// Return TriMesh of sphere embedding
TriMesh embedding_to_mesh(
        const TriMesh& _mesh_orig,
        const ExternalProperty<VH, Vec3d>& _embedding);

std::vector<TriMesh> lifted_meshes_from_mapstate(
        const MapState& _map_state);

/// Return an orthogonal vector to input vector
Vec3d any_orthogonal(
        const Vec3d& _v);

void compute_local_bases(
        const ExternalProperty<VH, Vec3d>& _embedding_input,
        ExternalProperty<VH, Vec3d>& _B1, // maxbe external property?
        ExternalProperty<VH, Vec3d>& _B2,
        const TriMesh& _mesh_input);

/// Compute local coordinates for input points, coordinate system centered in a
template <typename T>
void to_local_coordinates(
        const Vec3<T>& _a_in,
        const Vec3<T>& _b_in,
        const Vec3<T>& _c_in,
        Vec2<T>& _a_out,
        Vec2<T>& _b_out,
        Vec2<T>& _c_out)
{
    Vec3<T> normal = ((_b_in - _a_in).cross(_c_in - _a_in)).normalized();
    Vec3<T> basis0 = (_b_in - _a_in).normalized();
    Vec3<T> basis1 = normal.cross(basis0);

    _a_out = Vec2<T>(0.0, 0.0);
    _b_out = Vec2<T>((_b_in - _a_in).dot(basis0), 0.0);
    _c_out = Vec2<T>((_c_in - _a_in).dot(basis0), (_c_in - _a_in).dot(basis1));
}

/// Return number of flipped triangles on sphere with volume calculation
int n_flipped_volume(const MapState& _map_state);

/// compute length of the diagonal of the axis aligned bounding box of the mesh
double length_diagonal_of_bounding_box(const TriMesh& _mesh);

/// Checkerboard texture coordinates taken from T's SPHERE EMBEDDING, as lat-long
/// (phi, theta), rather than from a projection onto a world axis.
///
/// Why: the frontal projection builds texcoords as (x, y) of the lifted position, which
/// is a planar projection and therefore degenerates on every face parallel to the
/// projection axis. On the L -- a plate extruded along z -- that is 56% of the area:
/// measured UV foreshortening 0.014-0.037 on the extrusion walls against 0.98 on the
/// front and back, with 33.6% of the area at an effectively ZERO UV area. A collapsed UV
/// triangle makes the GPU's texture derivatives singular, which is what shows up as
/// shimmering on the walls and squashed checkers on the sides. No axis is a good choice:
/// the L has 12-23% of its area facing each of the six directions.
///
/// The sphere is the map's own shared domain, so a pattern laid out there is intrinsic:
/// it is non-degenerate everywhere except at the two poles, and "matching checkers means
/// matching material" still holds because the caller applies ONE texcoord array -- keyed
/// by halfedge of mesh_T -- to every lifted mesh.
///
/// Two details that matter:
///  - The phi SEAM. phi = atan2(y, x) jumps by 2*pi across the +-pi meridian, and a
///    triangle straddling it would interpolate the long way round and smear the whole
///    texture along that line. Texcoords are per-HALFEDGE precisely so this is fixable:
///    each face unwraps its own three phi values relative to its first vertex, so the
///    seam costs nothing.
///  - The POLES stay degenerate; phi is undefined there and the checkers pinch. That is
///    inherent to lat-long, but it is two small caps instead of 56% of the surface.
///
/// Aspect: u spans 2*_texture_factor over a full wrap and v spans _texture_factor, which
/// is the 2:1 ratio that keeps the checkers square.
ExternalProperty<HEH, Vec2d> sphere_embedding_texcoords(
        const TriMesh& _mesh_T,
        const ExternalProperty<VH, Vec3d>& _embedding_T,
        const double _texture_factor);

/// Make an existing per-halfedge texcoord field continuous across the phi seam, by
/// shifting each face's u values into the branch of its first vertex.
///
/// Needed because a texcoord TRANSFER produces per-vertex values: the target's texcoord
/// is looked up by heh.to() alone (see view_texture_frontal_projection_input), so a
/// vertex whose faces straddle the seam ends up with one branch for all of them, and the
/// faces on the other side then interpolate the long way round -- the checkers compress
/// across the whole texture along that one line. Re-unwrapping per face afterwards fixes
/// it, because the storage is per halfedge even when the values were computed per vertex.
///
/// _u_period is the u-distance of a full wrap, i.e. 2 * _texture_factor for
/// sphere_embedding_texcoords. Shifts are whole multiples of it, so with a tiling
/// texture the sampled texel is unchanged and only the interpolation is repaired.
void unwrap_texcoord_seam(
        const TriMesh& _mesh,
        ExternalProperty<HEH, Vec2d>& _texcoords,
        const double _u_period);


/// Checkerboard texture coordinates by PER-FACE DOMINANT-AXIS projection ("triplanar"):
/// each face is projected onto whichever of the three coordinate planes it faces most
/// directly.
///
/// Why this is the readable one. UV density (UV area per unit surface area) is exactly
/// |n_k| for the chosen axis k, and for a unit normal max_k |n_k| >= 1/sqrt(3), so the
/// density is bounded in [0.577, 1.0] -- it CANNOT collapse, and it varies by at most
/// 1.73x across any mesh whatsoever. That is a guarantee from the construction, not a
/// property of a particular shape. Compare the two alternatives on the L:
///   - frontal projection: density down to 0, with 53.75% of the area collapsed;
///   - sphere lat-long:    no collapse, but a 51x density spread plus two pinched poles.
/// A bounded, nearly uniform density is what makes the checkers the same size everywhere,
/// so a cell that looks stretched on the target really is stretched by the map.
///
/// The cost, and it is a real one: the chart changes from face to face, so the pattern is
/// DISCONTINUOUS across a chart boundary -- cells line up in size but not in phase. On a
/// faceted shape the boundaries follow the creases, where a break in the pattern reads as
/// the geometry; on a smooth shape they fall in the middle of smooth regions and read as
/// seams. This is the cube-map trade: uniform cells everywhere, paid for with seams on a
/// set of curves.
///
/// Why it takes _positions instead of reading _mesh.point(). The chart must be a function
/// of the SOURCE geometry alone, so that two different target meshes pick the same chart
/// for corresponding material; and it must be chosen on the TARGET's connectivity, so
/// that a face's three texcoords are always in one chart. Hence the caller transfers the
/// source POSITION to each target vertex -- a position is continuous on the source, so
/// the per-vertex transfer in view_texture_*_input loses nothing -- and the chart is
/// picked here, afterwards. Choosing the chart on the source and transferring the texcoord
/// does NOT work: that discontinuity is not a periodic shift, so unwrap_texcoord_seam
/// cannot repair it, and a target face whose vertices landed in different charts would
/// interpolate between unrelated projections and produce garbage along every boundary.
///
/// Same scheme as normal_cluster_tex_coords in Viewer/MeshView.cc (written upstream, never
/// called), minus its rgb-quadrant tinting. That one projects the mesh's own points, so it
/// can texture a single mesh but cannot show a map.
ExternalProperty<HEH, Vec2d> triplanar_texcoords(
        const TriMesh& _mesh,
        const ExternalProperty<VH, Vec3d>& _positions,
        const double _texture_factor);

}
