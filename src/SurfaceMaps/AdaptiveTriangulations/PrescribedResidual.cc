/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */

#include <SurfaceMaps/AdaptiveTriangulations/PrescribedResidual.hh>

#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedJacobianField.hh>
#include <SurfaceMaps/Utils/Helpers.hh>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace SurfaceMaps
{

double weighted_percentile(
        const std::vector<std::pair<double, double>>& _sorted,
        const double _total_weight,
        const double _q)
{
    double acc = 0.0;
    for (const auto& vw : _sorted)
    {
        acc += vw.second;
        if (acc >= _q * _total_weight)
            return vw.first;
    }
    return _sorted.back().first;
}

void compute_metric_residual(
        const MapState& _map_state,
        ResidualStats& _stats,
        const int _pair_idx)
{
    _stats = ResidualStats();

    if (_map_state.prescribed_jacobians.empty())
        return;
    ISM_ASSERT_L(_pair_idx, (int)_map_state.pairs_map_distortion.size());

    // (deviation, area of the lifted A triangle), kept so we can take percentiles
    // and area-weighted means at the end.
    std::vector<std::pair<double, double>> devs;
    devs.reserve(_map_state.mesh_T.n_faces());
    double total_area = 0.0;

    const int mesh_A_idx = _map_state.pairs_map_distortion[_pair_idx].first;
    const int mesh_B_idx = _map_state.pairs_map_distortion[_pair_idx].second;

    for (auto fh : _map_state.mesh_T.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_map_state.mesh_T, fh, vh_a, vh_b, vh_c);

        // Each T vertex carries TWO sphere positions, one per input mesh, and the
        // optimizer moves both independently (OptimizeMap.cc:426). The map A -> B is
        // "drop the T vertex onto A through sphere A, drop the SAME T vertex onto B
        // through sphere B", so the two lifts must use different sphere positions --
        // exactly as eval_trianglepair_energy_triangle_T does.
        const Vec3d a_sphere_A = _map_state.embeddings_T[mesh_A_idx][vh_a];
        const Vec3d b_sphere_A = _map_state.embeddings_T[mesh_A_idx][vh_b];
        const Vec3d c_sphere_A = _map_state.embeddings_T[mesh_A_idx][vh_c];
        const Vec3d a_sphere_B = _map_state.embeddings_T[mesh_B_idx][vh_a];
        const Vec3d b_sphere_B = _map_state.embeddings_T[mesh_B_idx][vh_b];
        const Vec3d c_sphere_B = _map_state.embeddings_T[mesh_B_idx][vh_c];

        auto lift = [&] (const Vec3d& _p, const int _idx)
        {
            return lift_vertex_to_surface(_p, _map_state.meshes_input[_idx],
                _map_state.meshes_embeddings_input[_idx], _map_state.bsp_embeddings_input[_idx]);
        };

        const Vec3d a_lifted_A = lift(a_sphere_A, mesh_A_idx);
        const Vec3d b_lifted_A = lift(b_sphere_A, mesh_A_idx);
        const Vec3d c_lifted_A = lift(c_sphere_A, mesh_A_idx);
        const Vec3d a_lifted_B = lift(a_sphere_B, mesh_B_idx);
        const Vec3d b_lifted_B = lift(b_sphere_B, mesh_B_idx);
        const Vec3d c_lifted_B = lift(c_sphere_B, mesh_B_idx);

        Eigen::Vector2d a_local_A, b_local_A, c_local_A, a_local_B, b_local_B, c_local_B;
        to_local_coordinates(a_lifted_A, b_lifted_A, c_lifted_A, a_local_A, b_local_A, c_local_A);
        to_local_coordinates(a_lifted_B, b_lifted_B, c_lifted_B, a_local_B, b_local_B, c_local_B);

        Eigen::Matrix2d M_A;
        M_A << b_local_A - a_local_A, c_local_A - a_local_A;
        Eigen::Matrix2d M_B;
        M_B << b_local_B - a_local_B, c_local_B - a_local_B;

        if (M_A.determinant() <= 0 || M_B.determinant() <= 0)
        {
            ++_stats.n_faces_skipped;
            continue;
        }

        const Eigen::Matrix2d J_actual = M_B * M_A.inverse();

        // Metric form of the prescription: only V and sigma, U is irrelevant.
        // A-side positions: the prescribed field is indexed by faces of mesh A.
        Eigen::Matrix2d V_local;
        Eigen::Vector2d sigma;
        lookup_prescribed_metric(a_sphere_A, b_sphere_A, c_sphere_A,
                                 a_lifted_A, b_lifted_A, c_lifted_A,
                                 _pair_idx, _map_state, V_local, sigma);

        if (sigma.minCoeff() <= 1e-6)
        {
            ++_stats.n_faces_skipped;
            continue;
        }

        // U drops out of singular values, so J V Sigma^-1 has the same spectrum
        // as the full residual J J*^-1.
        const Eigen::Matrix2d residual =
                J_actual * V_local * Eigen::Vector2d(1.0 / sigma[0], 1.0 / sigma[1]).asDiagonal();
        const Eigen::Vector2d s = Eigen::JacobiSVD<Eigen::Matrix2d>(residual).singularValues();

        // Both determinants are > 0 and sigma is bounded away from 0 (checked above), so
        // the residual is nonsingular; the clamp only guards against roundoff in the log.
        const double s0 = s[0];
        const double s1 = std::max(s[1], 1e-12);

        // det(M_A) = 2 * area of the lifted A triangle, i.e. the amount of mesh-A
        // material this T face is responsible for.
        const double area_A = 0.5 * M_A.determinant();

        devs.emplace_back((std::abs(s0 - 1.0) + std::abs(s1 - 1.0)) / 2.0, area_A);
        total_area += area_A;

        _stats.max = std::max(_stats.max, std::max(std::abs(s0 - 1.0), std::abs(s1 - 1.0)));
        _stats.area_part_aw  += area_A * std::abs(std::log(s0 * s1));
        _stats.shape_part_aw += area_A * std::abs(std::log(s0 / s1));
    }

    if (devs.empty())
        return;

    const int n = (int)devs.size();
    _stats.n_faces = n;
    _stats.area_measured = total_area;

    for (const auto& vw : devs)
    {
        _stats.mean += vw.first;
        _stats.mean_aw += vw.second * vw.first;
        if (vw.first > 0.1)
        {
            _stats.frac_gt_0p1 += 1.0;
            _stats.frac_gt_0p1_aw += vw.second;
        }
    }
    _stats.mean /= (double)n;
    _stats.frac_gt_0p1 /= (double)n;

    if (total_area > 0.0)
    {
        _stats.mean_aw /= total_area;
        _stats.frac_gt_0p1_aw /= total_area;
        _stats.area_part_aw /= total_area;
        _stats.shape_part_aw /= total_area;
    }

    std::sort(devs.begin(), devs.end()); // by deviation
    _stats.median = devs[n / 2].first;
    _stats.p90 = devs[std::min(n - 1, (int)(0.9 * n))].first;
    _stats.median_aw = weighted_percentile(devs, total_area, 0.5);
}

}
