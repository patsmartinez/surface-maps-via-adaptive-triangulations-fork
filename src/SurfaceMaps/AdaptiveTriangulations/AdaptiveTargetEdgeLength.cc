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


ExternalProperty<VH, double> map_driven_target_edge_lengths(
        const TriMesh& _mesh_A,
        const ExternalProperty<FH, PrescribedJacobian>& _prescribed,
        const double _jump_threshold,
        const double _floor,
        const double _gradation,
        const int _min_rings,
        int& _out_n_seam_verts)
{
    ISM_ASSERT_G(_floor, 0.0);
    ISM_ASSERT_GEQ(_gradation, 0.0);
    ISM_ASSERT_GEQ(_min_rings, 0);

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

}
