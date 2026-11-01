/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */

#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>

#include <TinyAD/Utils/Helpers.hh>
#include <SurfaceMaps/Utils/IO.hh>

namespace SurfaceMaps
{

// Return TriMesh of sphere embedding
TriMesh embedding_to_mesh(
        const TriMesh& _mesh_orig,
        const ExternalProperty<VH, Vec3d>& _embedding)
{
    TriMesh embedded_mesh;
    embedded_mesh.assign(_mesh_orig);
    for (auto v : embedded_mesh.vertices())
    {
        embedded_mesh.set_point(v, _embedding[v]);
    }
    return embedded_mesh;
}

std::vector<TriMesh> lifted_meshes_from_mapstate(
        const MapState& _map_state)
{
    const int n_meshes = _map_state.meshes_input.size();

    std::vector<TriMesh> lifted_Ts(n_meshes);
    for (int i = 0; i < n_meshes; ++i)
    {
        lifted_Ts[i] = embedding_to_mesh(
                    _map_state.mesh_T,
                    lift_to_surface(
                        _map_state.meshes_input[i],
                        _map_state.mesh_T,
                        _map_state.embeddings_input[i],
                        _map_state.embeddings_T[i],
                        _map_state.bsp_embeddings_input[i]));
    }

    return lifted_Ts;
}

/// Compute an abitrary tangent vector of the sphere at position p.
Eigen::Vector3d any_orthogonal(
        const Eigen::Vector3d& _p)
{
    // Find coordinate axis spanning the largest angle with _p.
    // Return cross product that of axis with _p
    Eigen::Vector3d tang;
    double min_abs_dot = INFINITY;
    for (const Eigen::Vector3d& ax : { Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0) })
    {
        double abs_dot = fabs(_p.dot(ax));
        if (abs_dot < min_abs_dot)
        {
            min_abs_dot = abs_dot;
            tang = ax.cross(_p).normalized();
        }
    }

    return tang;
}

void compute_local_bases(
        const ExternalProperty<VH, Vec3d>& _embedding_input,
        ExternalProperty<VH, Vec3d>& _B1,
        ExternalProperty<VH, Vec3d>& _B2,
        const TriMesh& _mesh_input)
{
    for (auto vh : _mesh_input.vertices())
    {
        _B1[vh] = any_orthogonal(_embedding_input[vh]);
        _B2[vh] = _embedding_input[vh].cross(_B1[vh]);
    }
}


/// Return number of flipped triangles on sphere with volume calculation
int n_flipped_volume(const MapState& _map_state)
{
    int n_flipped = 0;
    for(auto fh : _map_state.mesh_T.faces())
    {
        auto a_idx = fh.halfedge().to();
        auto b_idx = fh.halfedge().next().to();
        auto c_idx = fh.halfedge().from();
        for (int i = 0; i < (int)_map_state.meshes_input.size(); i++)
        {
            Eigen::Vector3<double> a_mani = _map_state.embeddings_T[i][a_idx];
            Eigen::Vector3<double> b_mani = _map_state.embeddings_T[i][b_idx];
            Eigen::Vector3<double> c_mani = _map_state.embeddings_T[i][c_idx];

            const double volume = TinyAD::col_mat(a_mani, b_mani, c_mani).determinant();
            if (volume <= 0.0)
            {
                n_flipped++;
                break;
            }
        }
    }
    return n_flipped;
}

double length_diagonal_of_bounding_box(const TriMesh& _mesh)
{
    MatXd V;
    MatXi F;
    mesh_to_matrix(_mesh, V, F);
    // bounding box points
    Eigen::Vector3d min_point = V.colwise().minCoeff();
    Eigen::Vector3d max_point = V.colwise().maxCoeff();

    return (min_point-max_point).norm();
}

ExternalProperty<HEH, Vec2d> sphere_embedding_texcoords(
        const TriMesh& _mesh_T,
        const ExternalProperty<VH, Vec3d>& _embedding_T,
        const double _texture_factor)
{
    ExternalProperty<HEH, Vec2d> texcoords(_mesh_T);

    // phi in (-pi, pi], theta in [0, pi]; scaled so u spans 2*factor and v spans factor.
    auto lat_long = [&] (const VH _vh, double& _phi, double& _theta)
    {
        const Vec3d p = _embedding_T[_vh].normalized();
        _phi = std::atan2(p[1], p[0]);
        _theta = std::acos(std::clamp(p[2], -1.0, 1.0));
    };

    for (auto fh : _mesh_T.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_mesh_T, fh, vh_a, vh_b, vh_c);

        double phi_a, theta_a, phi_b, theta_b, phi_c, theta_c;
        lat_long(vh_a, phi_a, theta_a);
        lat_long(vh_b, phi_b, theta_b);
        lat_long(vh_c, phi_c, theta_c);

        // Unwrap b and c into the branch of a, so a face straddling the +-pi meridian
        // gets continuous phi instead of a jump of nearly 2*pi.
        const auto unwrap = [&] (double& _phi)
        {
            while (_phi - phi_a >  M_PI) _phi -= 2.0 * M_PI;
            while (_phi - phi_a < -M_PI) _phi += 2.0 * M_PI;
        };
        unwrap(phi_b);
        unwrap(phi_c);

        const double s = _texture_factor / M_PI;
        const Vec2d uv_a(phi_a * s, theta_a * s);
        const Vec2d uv_b(phi_b * s, theta_b * s);
        const Vec2d uv_c(phi_c * s, theta_c * s);

        // Per-halfedge, so each face keeps its own unwrapped values. A halfedge's
        // texcoord belongs to the vertex it points TO, matching the frontal-projection
        // convention in Visualization.cc.
        for (auto heh : _mesh_T.fh_range(fh))
        {
            const VH to = heh.to();
            if (to == vh_a)      texcoords[heh] = uv_a;
            else if (to == vh_b) texcoords[heh] = uv_b;
            else                 texcoords[heh] = uv_c;
        }
    }

    return texcoords;
}

void unwrap_texcoord_seam(
        const TriMesh& _mesh,
        ExternalProperty<HEH, Vec2d>& _texcoords,
        const double _u_period)
{
    if (_u_period <= 0.0)
        return;

    const double half = 0.5 * _u_period;
    for (auto fh : _mesh.faces())
    {
        HEH heh_a, heh_b, heh_c;
        handles(_mesh, fh, heh_a, heh_b, heh_c);

        const double u_ref = _texcoords[heh_a][0];
        for (HEH heh : { heh_b, heh_c })
        {
            double u = _texcoords[heh][0];
            while (u - u_ref >  half) u -= _u_period;
            while (u - u_ref < -half) u += _u_period;
            _texcoords[heh] = Vec2d(u, _texcoords[heh][1]);
        }
    }
}


ExternalProperty<HEH, Vec2d> triplanar_texcoords(
        const TriMesh& _mesh,
        const ExternalProperty<VH, Vec3d>& _positions,
        const double _texture_factor)
{
    ExternalProperty<HEH, Vec2d> texcoords(_mesh);

    for (auto fh : _mesh.faces())
    {
        HEH heh_a, heh_b, heh_c;
        handles(_mesh, fh, heh_a, heh_b, heh_c);

        const Vec3d p_a = _positions[_mesh.to_vertex_handle(heh_a)];
        const Vec3d p_b = _positions[_mesh.to_vertex_handle(heh_b)];
        const Vec3d p_c = _positions[_mesh.to_vertex_handle(heh_c)];

        // Normal of the triangle as it sits on the SOURCE, which is what makes two
        // different target meshes agree on the chart. Left unnormalized on purpose:
        // argmax|n_k| is unchanged by a positive scale, so this needs no length and is
        // safe on a tiny triangle.
        const Vec3d n = (p_b - p_a).cross(p_c - p_a);

        // Relative tolerance, so an exact tie resolves to the lowest axis index EVERY
        // time. Without it, a facet whose normal ties two axes -- the L's chamfers are at
        // (+-2/3, +-1/3, +-2/3) and (+-1/sqrt2, 0, +-1/sqrt2), so |n_x| == |n_z| exactly
        // -- has its triangles split between two charts by nothing but rounding in the
        // cross product above, which puts a seam through the middle of a FLAT face where
        // the geometry gives no reason for one. Measured on the L: that accounts for 26%
        // of all chart-boundary length, and the tolerance removes every bit of it (total
        // boundary 7.57% -> 5.68% of edge length, and what remains is 100% on creases).
        // Any eps from 1e-12 to 1e-3 gives the identical result, so this is not a tuned
        // threshold; it just has to clear double rounding noise.
        constexpr double eps = 1e-9;
        int axis = 0;
        if (std::fabs(n[1]) > std::fabs(n[axis]) * (1.0 + eps)) axis = 1;
        if (std::fabs(n[2]) > std::fabs(n[axis]) * (1.0 + eps)) axis = 2;
        // A degenerate source triangle gives n == 0 and falls through to axis 0. Its UV
        // is then degenerate too, but so is the triangle; there is nothing to recover.

        // The remaining two axes, in cyclic order, so the chart stays right-handed on the
        // +axis side. It is mirrored on the -axis side, which a checkerboard cannot show.
        const int u_axis = (axis + 1) % 3;
        const int v_axis = (axis + 2) % 3;

        for (HEH heh : { heh_a, heh_b, heh_c })
        {
            const Vec3d p = _positions[_mesh.to_vertex_handle(heh)];
            texcoords[heh] = Vec2d(p[u_axis] * _texture_factor, p[v_axis] * _texture_factor);
        }
    }

    return texcoords;
}

}
