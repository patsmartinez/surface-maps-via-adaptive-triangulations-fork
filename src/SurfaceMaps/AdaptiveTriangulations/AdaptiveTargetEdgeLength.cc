/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Dörte Pieper, Patrick Schmidt
 */

#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTargetEdgeLength.hh>

#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/Utils/PrincipalCurvature.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Misc/Diffusion.hh>
#include <SurfaceMaps/Viewer/HeatmapColors.hh>
#include <SurfaceMaps/Viewer/Colors.hh>

#include <queue>

namespace SurfaceMaps
{

ExternalProperty<VH, double> target_edge_lengths(
        const TriMesh& _mesh,
        const double _approx_error,
        const double _min_edge_length,
        const double _max_edge_length)
{
    ISM_ASSERT_G(_approx_error, 0.0);
    ISM_ASSERT_L(_min_edge_length, _max_edge_length);

    const auto max_abs_curv = max_abs_curvatures(_mesh);

    ExternalProperty<VH, double> tels(_mesh);
    for (auto v : _mesh.vertices())
    {
        tels[v] = sqrt(6.0 * _approx_error / max_abs_curv[v] - 3.0 * sqr(_approx_error));
        tels[v] = std::clamp(tels[v], _min_edge_length, _max_edge_length);
        if(isnan(tels[v]))
            tels[v] = _min_edge_length;
    }

    return tels;
}


double mean_edge_length(
        const TriMesh& _mesh)
{
    double sum = 0.0;
    int n = 0;
    for (auto eh : _mesh.edges())
    {
        sum += (_mesh.point(eh.v0()) - _mesh.point(eh.v1())).norm();
        ++n;
    }
    return n > 0 ? sum / n : 0.0;
}


double map_tel_auto_diffusion_length(
        const TriMesh& _mesh_A,
        const double _floor)
{
    return std::max(2.0 * _floor, mean_edge_length(_mesh_A));
}


ExternalProperty<VH, double> map_driven_target_edge_lengths(
        const TriMesh& _mesh_A,
        const ExternalProperty<FH, PrescribedJacobian>& _prescribed,
        const double _jump_threshold,
        const double _floor,
        const double _far_tel,
        const MapTelMethod _method,
        const double _diffusion_length,
        const double _gradation,
        const int _min_rings,
        int& _out_n_seam_verts,
        ExternalProperty<VH, double>* _out_jump,
        ExternalProperty<VH, double>* _out_phi)
{
    ISM_ASSERT_G(_floor, 0.0);
    ISM_ASSERT_G(_far_tel, _floor);
    ISM_ASSERT_GEQ(_gradation, 0.0);
    ISM_ASSERT_GEQ(_min_rings, 0);
    ISM_ASSERT_GEQ(_diffusion_length, 0.0);

    // Per-vertex jump magnitude: the LARGEST prescribed discontinuity on any incident
    // edge. Max, not average -- see the header.
    ExternalProperty<VH, double> jump(_mesh_A, 0.0);
    for (auto eh : _mesh_A.edges())
    {
        if (eh.is_boundary())
            continue;

        const Eigen::Vector2d& s_a = _prescribed[eh.h0().face()].sigma;
        const Eigen::Vector2d& s_b = _prescribed[eh.h0().opp().face()].sigma;
        if (s_a.minCoeff() <= 0.0 || s_b.minCoeff() <= 0.0)
            continue;

        const double d = std::sqrt(sqr(std::log(s_a[0] / s_b[0])) + sqr(std::log(s_a[1] / s_b[1])));
        jump[eh.v0()] = std::max(jump[eh.v0()], d);
        jump[eh.v1()] = std::max(jump[eh.v1()], d);
    }

    if (_out_jump)
        *_out_jump = jump;

    ExternalProperty<VH, bool> in_seam(_mesh_A, false);
    std::vector<VH> seam;
    for (auto vh : _mesh_A.vertices())
    {
        if (jump[vh] > _jump_threshold)
        {
            in_seam[vh] = true;
            seam.push_back(vh);
        }
    }
    _out_n_seam_verts = (int)seam.size();

    // An empty seam is a no-op either way: INF everywhere, so the caller's min() leaves
    // the curvature field untouched. Returned early because the diffusion below
    // normalizes by its own maximum, which would be zero here.
    if (seam.empty())
        return ExternalProperty<VH, double>(_mesh_A, INF_DOUBLE);

    if (_method == MapTelMethod::HeatDiffusion)
    {
        // SHORT-TIME HEAT FLOW from the seam. One implicit-Euler step,
        //     (M - t L) u = M u0,     u0 = 1 on the seam and 0 elsewhere,
        // with the cotan Laplacian L and lumped mass M (Diffusion.cc), then
        //     phi = u / max(u)  in [0, 1],
        //     tel = far + (floor - far) * phi.
        //
        // Why this rather than dilate-then-Dijkstra:
        //
        //  - The thickness is a LENGTH, not a ring count. t = w^2, so u falls off over
        //    w world units whatever the input tessellation is. The ring dilation it
        //    replaces meant something different on every mesh -- one ring is 0.0629 on
        //    the L and 0.0149 on the U, a 4x different band for the same rule.
        //
        //  - It is C^infinity away from the seam instead of C^0. Dijkstra-over-edges is
        //    a piecewise-linear function of the vertex positions with a crease at every
        //    equidistant point, and it also overestimates true geodesic distance by up
        //    to ~15% on an irregular mesh because paths can only run along edges. Heat
        //    flow sees the surface, not the edge graph.
        //
        //  - The fall-off is isotropic in SURFACE metric, so the band follows the
        //    geometry (it narrows across a thin feature, widens on a flat sheet) rather
        //    than following the connectivity. That is the sense in which the thickness
        //    is physical, and it is what should matter on models more complex than these.
        //
        // The field is deliberately NOT a distance: no Varadhan log is taken. Varadhan
        // is exact only as t -> 0 and is numerically poor at any usable t, and an
        // accurate geodesic is not what is wanted here -- a smooth monotone fall-off
        // with a controlled length scale is. phi is already that, and it makes w the
        // whole thickness knob, so the gradation parameter has no role on this path.
        // (Supported by the measurements behind map_tel_gradation: cost varied only
        // 3-12% from g = 0.25 to infinity, i.e. the gradation never mattered.)
        //
        // The far value is the caller's max_edge_length, which the curvature field is
        // itself clamped to, so the two agree exactly where phi -> 0.
        // Auto width = max(2 * floor, mean edge).
        //
        // NOT the textbook t = m*h^2 heuristic. That is tuned for recovering geodesic
        // distance, where h is the only scale in the problem; here there is a better one.
        // The transition should span about two of the elements the field is ASKING for,
        // which ties w to _floor and makes it independent of the input tessellation --
        // the whole point of the change. Measured on the L (floor 0.015): 2*h = 0.089
        // gives a band over 39.6% of A's area, while 2*floor = 0.03 gives 19.1%.
        //
        // Clamped below by the mean edge because a diffusion length under one element is
        // not representable: the implicit step smears by about h regardless, so a smaller
        // t only buys arithmetic. That bound binds exactly when the floor is finer than
        // the input mesh, which map_tel_floor's own documentation says buys nothing
        // anyway.
        double w = _diffusion_length;
        if (w <= 0.0)
            w = map_tel_auto_diffusion_length(_mesh_A, _floor);
        ISM_ASSERT_G(w, 0.0);

        ExternalProperty<VH, double> u(_mesh_A, 0.0);
        for (const VH vh : seam)
            u[vh] = 1.0;

        diffuse_pointwise_field(_mesh_A, u, w * w);

        double u_max = 0.0;
        for (auto vh : _mesh_A.vertices())
            u_max = std::max(u_max, u[vh]);
        ISM_ASSERT_G(u_max, 0.0);

        if (_out_phi)
            *_out_phi = ExternalProperty<VH, double>(_mesh_A, 0.0);

        // Below this, tel sits within 0.1% of the curvature cap, i.e. it asks for nothing
        // the curvature field was not already going to do. Left at INF instead, so the
        // no-op contract outside the band stays exact.
        //
        // Note this is NOT the band edge: heat has exponential tails, so phi stays above
        // 1e-3 out to ~5w, and on a model whose extent is only ~10w that is most of the
        // surface at a tel barely under the cap. The band is where tel is actually small
        // -- which is what the "band" figure in the caller's log reports, and what
        // n_band_T in the app counts.
        constexpr double phi_eps = 1e-3;

        ExternalProperty<VH, double> tel(_mesh_A, INF_DOUBLE);
        for (auto vh : _mesh_A.vertices())
        {
            // Clamped because an obtuse triangle gives a negative cotan weight, which
            // can push u slightly outside [0, u_max].
            const double phi = std::clamp(u[vh] / u_max, 0.0, 1.0);
            if (_out_phi)
                (*_out_phi)[vh] = phi;

            if (phi >= phi_eps)
                tel[vh] = _far_tel + (_floor - _far_tel) * phi;
        }

        // phi is 1 only at the single hottest vertex, so pin the seam itself to the
        // floor and keep the guarantee the Dijkstra path gave.
        for (const VH vh : seam)
            tel[vh] = _floor;

        return tel;
    }

    // ---- MapTelMethod::GeodesicDijkstra, the original path ------------------------
    // Dilate, so the band is wide enough for the centroid sampler to find it.
    for (int ring = 0; ring < _min_rings; ++ring)
    {
        std::vector<VH> added;
        for (const VH vh : seam)
        {
            for (auto vv : _mesh_A.vv_range(vh))
            {
                if (!in_seam[vv])
                {
                    in_seam[vv] = true;
                    added.push_back(vv);
                }
            }
        }
        seam.insert(seam.end(), added.begin(), added.end());
    }

    // tel = floor + g * dist(seam), Dijkstra over mesh edges. INF where unreached, so the
    // caller's min() leaves the curvature field alone outside the band.
    ExternalProperty<VH, double> tel(_mesh_A, INF_DOUBLE);
    ExternalProperty<VH, double> dist(_mesh_A, INF_DOUBLE);
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>,
                        std::greater<std::pair<double, int>>> queue;
    for (const VH vh : seam)
    {
        dist[vh] = 0.0;
        tel[vh] = _floor;
        queue.push({ 0.0, vh.idx() });
    }

    while (!queue.empty())
    {
        const double d = queue.top().first;
        const VH vh(queue.top().second);
        queue.pop();
        if (d > dist[vh])
            continue;

        for (auto vv : _mesh_A.vv_range(vh))
        {
            const double nd = d + (_mesh_A.point(vv) - _mesh_A.point(vh)).norm();
            if (nd < dist[vv])
            {
                dist[vv] = nd;
                tel[vv] = _floor + _gradation * nd;
                queue.push({ nd, vv.idx() });
            }
        }
    }

    return tel;
}


void write_vertex_field_ply(
        const TriMesh& _mesh,
        const ExternalProperty<VH, double>& _values,
        const fs::path& _path,
        const double _min,
        const double _max)
{
    // Per FACE, the mean of its three vertices -- which is exactly the value
    // ideal_triangle_edge_length's centroid sample interpolates, so the picture shows
    // what the optimizer reads and not what the field holds between samples.
    ExternalProperty<FH, double> per_face(_mesh, 0.0);
    double lo = INF_DOUBLE;
    double hi = -INF_DOUBLE;
    for (auto fh : _mesh.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_mesh, fh, vh_a, vh_b, vh_c);
        double sum = 0.0;
        int n_finite = 0;
        for (const VH vh : { vh_a, vh_b, vh_c })
        {
            if (std::isfinite(_values[vh]))
            {
                sum += _values[vh];
                ++n_finite;
            }
        }
        // All-INF faces are "outside the band"; park them at the top of the range.
        per_face[fh] = n_finite > 0 ? sum / n_finite : INF_DOUBLE;
        if (std::isfinite(per_face[fh]))
        {
            lo = std::min(lo, per_face[fh]);
            hi = std::max(hi, per_face[fh]);
        }
    }

    double vmin = _min;
    double vmax = _max;
    if (vmin == 0.0 && vmax == 0.0)
    {
        vmin = std::isfinite(lo) ? lo : 0.0;
        vmax = std::isfinite(hi) ? hi : 1.0;
    }
    if (vmax <= vmin)
        vmax = vmin + 1.0;

    for (auto fh : _mesh.faces())
    {
        if (!std::isfinite(per_face[fh]))
            per_face[fh] = vmax;
    }

    // MAGENTA at the low end, WHITE at the high end: on a sizing field that reads as
    // "coloured where it asks for refinement", on phi as "coloured where it is hot".
    const auto colors = linear_colors(per_face, vmin, vmax, MAGENTA, WHITE);
    write_colored_ply_flat(_mesh, colors, _path);
    ISM_INFO("Wrote " << _path << " (range " << vmin << " .. " << vmax << ")");
}

}
