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

}
