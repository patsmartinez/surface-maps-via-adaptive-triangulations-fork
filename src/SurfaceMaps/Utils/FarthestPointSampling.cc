/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */

#include "FarthestPointSampling.hh"

#include <SurfaceMaps/Utils/Out.hh>

#include <queue>

namespace SurfaceMaps
{

std::vector<VH> farthest_point_sampling(
        const TriMesh& _mesh,
        const int _n_samples)
{
    ISM_ASSERT_G(_n_samples, 0);
    ISM_ASSERT_LEQ(_n_samples, (int)_mesh.n_vertices());

    // Deterministic seed: vertex farthest from the centroid (i.e. an extremity)
    Vec3d centroid(0.0, 0.0, 0.0);
    for (auto vh : _mesh.vertices())
        centroid += _mesh.point(vh);
    centroid /= (double)_mesh.n_vertices();

    VH vh_seed(0);
    double dist_seed = -1.0;
    for (auto vh : _mesh.vertices())
    {
        const double d = (_mesh.point(vh) - centroid).squaredNorm();
        if (d > dist_seed)
        {
            dist_seed = d;
            vh_seed = vh;
        }
    }

    // dist[v] = distance to the nearest sample chosen so far
    std::vector<double> dist(_mesh.n_vertices(), INF_DOUBLE);
    std::vector<VH> samples;

    // Add a sample and propagate the distance improvements it causes (Dijkstra)
    auto add_sample = [&] (const VH _vh)
    {
        samples.push_back(_vh);

        using QueueEntry = std::pair<double, int>; // (distance, vertex index)
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;

        dist[_vh.idx()] = 0.0;
        queue.push({ 0.0, _vh.idx() });

        while (!queue.empty())
        {
            const double d = queue.top().first;
            const VH vh(queue.top().second);
            queue.pop();

            // Lazy deletion: skip outdated entries
            if (d > dist[vh.idx()])
                continue;

            for (auto heh : _mesh.voh_range(vh))
            {
                const VH vh_to = _mesh.to_vertex_handle(heh);
                const double d_to = d + _mesh.calc_edge_length(heh);
                if (d_to < dist[vh_to.idx()])
                {
                    dist[vh_to.idx()] = d_to;
                    queue.push({ d_to, vh_to.idx() });
                }
            }
        }
    };

    add_sample(vh_seed);

    while ((int)samples.size() < _n_samples)
    {
        // Pick the vertex farthest from all current samples
        int idx_best = 0;
        for (int i = 1; i < (int)dist.size(); ++i)
        {
            if (dist[i] > dist[idx_best])
                idx_best = i;
        }

        add_sample(VH(idx_best));
    }

    return samples;
}

}
