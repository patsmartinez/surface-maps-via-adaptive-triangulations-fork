/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */

#include "PrescribedJacobianField.hh"

#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>

namespace SurfaceMaps
{

ExternalProperty<FH, PrescribedJacobian> extract_jacobian_field(
        const TriMesh& mesh_A,
        const TriMesh& mesh_B)
{
    ISM_ASSERT_EQ(mesh_A.n_faces(), mesh_B.n_faces());
    ISM_ASSERT_EQ(mesh_A.n_vertices(), mesh_B.n_vertices());

    ExternalProperty<FH, PrescribedJacobian> result(mesh_A);

    for (auto fh : mesh_A.faces())
    {
        // Get vertex handles
        VH vh_a, vh_b, vh_c;
        handles(mesh_A, fh, vh_a, vh_b, vh_c);

        // Get 3D positions on both meshes
        Vec3d a_A = mesh_A.point(vh_a);
        Vec3d b_A = mesh_A.point(vh_b);
        Vec3d c_A = mesh_A.point(vh_c);

        Vec3d a_B = mesh_B.point(vh_a);
        Vec3d b_B = mesh_B.point(vh_b);
        Vec3d c_B = mesh_B.point(vh_c);

        // Compute local bases for mesh A
        Vec3d normal_A = ((b_A - a_A).cross(c_A - a_A)).normalized();
        Vec3d basis0_A = (b_A - a_A).normalized();
        Vec3d basis1_A = normal_A.cross(basis0_A);

        // Compute local bases for mesh B
        Vec3d normal_B = ((b_B - a_B).cross(c_B - a_B)).normalized();
        Vec3d basis0_B = (b_B - a_B).normalized();
        Vec3d basis1_B = normal_B.cross(basis0_B);

        // Convert to local 2D coordinates
        Eigen::Vector2d a_local_A, b_local_A, c_local_A;
        Eigen::Vector2d a_local_B, b_local_B, c_local_B;
        to_local_coordinates(a_A, b_A, c_A, a_local_A, b_local_A, c_local_A);
        to_local_coordinates(a_B, b_B, c_B, a_local_B, b_local_B, c_local_B);

        // Build edge matrices
        Eigen::Matrix2d M_A;
        M_A << b_local_A - a_local_A, c_local_A - a_local_A;
        Eigen::Matrix2d M_B;
        M_B << b_local_B - a_local_B, c_local_B - a_local_B;

        // Compute Jacobian J = M_B * M_A^{-1}
        Eigen::Matrix2d J = M_B * M_A.inverse();

        // Compute SVD: J = U * Sigma * V^T
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Store singular values
        result[fh].sigma = svd.singularValues();

        // Get 2D singular vectors
        Eigen::Vector2d v1_2d = svd.matrixV().col(0);
        Eigen::Vector2d v2_2d = svd.matrixV().col(1);
        Eigen::Vector2d u1_2d = svd.matrixU().col(0);
        Eigen::Vector2d u2_2d = svd.matrixU().col(1);

        // Lift V vectors to 3D (tangent to mesh A)
        result[fh].V.col(0) = v1_2d[0] * basis0_A + v1_2d[1] * basis1_A;
        result[fh].V.col(1) = v2_2d[0] * basis0_A + v2_2d[1] * basis1_A;

        // Lift U vectors to 3D (tangent to mesh B)
        result[fh].U.col(0) = u1_2d[0] * basis0_B + u1_2d[1] * basis1_B;
        result[fh].U.col(1) = u2_2d[0] * basis0_B + u2_2d[1] * basis1_B;
    }

    return result;
}

template <typename T>
Eigen::Matrix2<T> lookup_prescribed_jacobian(
        const Vec3<T>& _a_sphere,
        const Vec3<T>& _b_sphere,
        const Vec3<T>& _c_sphere,
        const Vec3<T>& _a_lifted_A,
        const Vec3<T>& _b_lifted_A,
        const Vec3<T>& _c_lifted_A,
        const Vec3<T>& _a_lifted_B,
        const Vec3<T>& _b_lifted_B,
        const Vec3<T>& _c_lifted_B,
        const int _pair_idx,
        const MapState& _map_state)
{
    // Get the mesh pair indices
    const int mesh_A_idx = _map_state.pairs_map_distortion[_pair_idx].first;

    // Compute centroid of T-triangle on sphere (use passive values for lookup)
    Vec3d centroid_sphere = (TinyAD::to_passive(_a_sphere) +
                             TinyAD::to_passive(_b_sphere) +
                             TinyAD::to_passive(_c_sphere)) / 3.0;
    centroid_sphere.normalize();

    // Find the face of mesh A's embedding that contains the centroid
    SFH containing_face;
    double alpha, beta, gamma;
    bsp_tree_barys_face(centroid_sphere,
                        _map_state.meshes_embeddings_input[mesh_A_idx],
                        _map_state.bsp_embeddings_input[mesh_A_idx],
                        alpha, beta, gamma, containing_face);

    // Get the prescribed Jacobian for this face
    const PrescribedJacobian& pj = _map_state.prescribed_jacobians[_pair_idx][containing_face];

    // Compute local basis for T-triangle on mesh A surface
    Vec3<T> normal_T_A = ((_b_lifted_A - _a_lifted_A).cross(_c_lifted_A - _a_lifted_A)).normalized();
    Vec3<T> basis0_T_A = (_b_lifted_A - _a_lifted_A).normalized();
    Vec3<T> basis1_T_A = normal_T_A.cross(basis0_T_A);

    // Compute local basis for T-triangle on mesh B surface
    Vec3<T> normal_T_B = ((_b_lifted_B - _a_lifted_B).cross(_c_lifted_B - _a_lifted_B)).normalized();
    Vec3<T> basis0_T_B = (_b_lifted_B - _a_lifted_B).normalized();
    Vec3<T> basis1_T_B = normal_T_B.cross(basis0_T_B);

    // Project V vectors (from mesh A) into T-triangle's local basis on A
    // V is 3x2, we need to project each column
    Eigen::Matrix2<T> V_local;
    for (int i = 0; i < 2; ++i)
    {
        Vec3d v_3d = pj.V.col(i);
        V_local(0, i) = basis0_T_A.dot(v_3d);
        V_local(1, i) = basis1_T_A.dot(v_3d);
    }

    // Project U vectors (from mesh B) into T-triangle's local basis on B
    Eigen::Matrix2<T> U_local;
    for (int i = 0; i < 2; ++i)
    {
        Vec3d u_3d = pj.U.col(i);
        U_local(0, i) = basis0_T_B.dot(u_3d);
        U_local(1, i) = basis1_T_B.dot(u_3d);
    }

    // Reconstruct J* = U * Sigma * V^T in T-triangle's local coordinates
    Eigen::DiagonalMatrix<T, 2> Sigma(T(pj.sigma[0]), T(pj.sigma[1]));
    Eigen::Matrix2<T> J_star = U_local * Sigma * V_local.transpose();

    return J_star;
}

// Explicit template instantiations
template Eigen::Matrix2<double> lookup_prescribed_jacobian(
        const Vec3<double>&, const Vec3<double>&, const Vec3<double>&,
        const Vec3<double>&, const Vec3<double>&, const Vec3<double>&,
        const Vec3<double>&, const Vec3<double>&, const Vec3<double>&,
        const int, const MapState&);

template Eigen::Matrix2<TinyAD::Double<12,false>> lookup_prescribed_jacobian(
        const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&,
        const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&,
        const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&, const Vec3<TinyAD::Double<12,false>>&,
        const int, const MapState&);

template Eigen::Matrix2<TinyAD::Double<12,true>> lookup_prescribed_jacobian(
        const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&,
        const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&,
        const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&, const Vec3<TinyAD::Double<12,true>>&,
        const int, const MapState&);

}
