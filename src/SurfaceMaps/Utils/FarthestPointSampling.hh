/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */
#pragma once

#include <SurfaceMaps/Types.hh>

namespace SurfaceMaps
{

/// Greedily sample vertices that are far apart, using edge-graph (Dijkstra) distances
/// as an approximation of geodesic distance.
/// The first sample is the vertex farthest from the mesh centroid, which makes the
/// result deterministic and starts the sampling at an extremity.
std::vector<VH> farthest_point_sampling(
        const TriMesh& _mesh,
        const int _n_samples);

}
