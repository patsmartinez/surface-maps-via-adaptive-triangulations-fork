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
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>

namespace SurfaceMaps
{

struct AdaptiveTriangulationsSettings
{
    // Schedule
    int max_iterations = 200;
    int optimize_per_remeshing_iters = 1;

    // Prescribed Jacobian mode
    bool use_prescribed_jacobian = false;
    bool prescribed_metric_form = true; // Metric form (V, sigma only, U ignored); false = full J* residual

    /// Make E_mesh prescription-aware instead of merely down-weighting it.
    ///
    /// E_mesh compares each lifted T triangle to an ideal EQUILATERAL of side tel, once
    /// per input mesh. Demanding equilateral on A *and* on B implies J is a rotation, so
    /// E_mesh carries a hidden isometry prior that is mutually unsatisfiable with an
    /// anisotropic prescription. With this on, the B-side ideal instead carries the
    /// prescribed anisotropy (see prescribed_ideal_shape), so satisfying both terms
    /// implies J ~ J* rather than J ~ rotation.
    ///
    /// Reduces EXACTLY to the old behaviour when J* = I, so A-on-A rows must stay
    /// bit-identical. Requires use_prescribed_jacobian.
    bool prescribed_mesh_ideal = false;

    // Weights
    double w_barrier = 1e-4;
    double w_reg = 0.0;
    double w_mesh = 1.0;
    double w_map = 1.0;
    double w_landmark_sphere = 0.0;
    double w_landmark_world = 0.0;
    double w_approx = 1.0;
    double w_bound = 0.0; // Surface approximation bound weight
    double surface_approx_bound = INFINITY;
    bool hard_landmark_constraints = false;
    double w_angle = 0.0;

    // Target edge length
    double approx_error = 0.0025;
    double min_edge_length = 0.0001;
    double max_edge_length = 0.1;

    // Solver settings
    /// Log the A<->B antisymmetric part of the gradient and of the Newton direction,
    /// once per iteration. Only meaningful on a SELF-MAP (both input meshes, embeddings
    /// and liftings identical), where the energy is provably symmetric under swapping
    /// the two embeddings, so a symmetric configuration must stay symmetric. Diagnostic
    /// only -- it changes nothing about the optimization.
    /// Treat E_mesh's target edge length as the SIZING FIELD it is, rather than as a
    /// differentiable function of the map.
    ///
    /// The mesh element picks mesh_tel_idx = argmin_i tel_i (T must be fine enough for
    /// every input mesh) using variables_passive, then RE-computes tel from that mesh's
    /// variables ACTIVELY. Two consequences:
    ///   - tel becomes a degree of freedom. Symmetric Dirichlet against an ideal of side
    ///     tel scales like (l/tel)^2, so d/d(tel) ~ -2 l^2 / tel^3, and the adaptive
    ///     sizing floor is 1e-3 -- hence a gradient of order 1e9. Measured on the U
    ///     self-map: |g| from E_mesh alone at w_mesh = 0.01 is 1.9e9, against 1.6 from
    ///     E_map and 17 from E_approx. The optimizer can lower E_mesh by inflating its
    ///     own TARGET instead of improving the triangle, and that mode outweighs
    ///     everything the map terms ask for by nine orders of magnitude.
    ///   - the argmin tie-break is strict `<`, so equal tels always select i = 0. On a
    ///     self-map every triangle therefore routes its tel through mesh 0, putting the
    ///     whole 1e9 gradient on mesh 0's half and none on mesh 1's. That is a gradient
    ///     86% ANTISYMMETRIC at a configuration where the two embeddings agree to
    ///     1.1e-16 -- which is how the self-map floor gets created, since a symmetric
    ///     energy at a symmetric point cannot move off it.
    ///
    /// With this on, tel is the value min_tel already computed passively: same number,
    /// no derivative. Off by default so the published algorithm stays reproducible.
    bool mesh_tel_passive = false;

    /// MAP-DRIVEN sizing field (prototype): also refine T where the PRESCRIBED metric
    /// jumps, not only where the surface curves. Combined with the curvature field by
    /// min, inside init_target_edge_lengths. See map_driven_target_edge_lengths.
    ///
    /// Needs mesh_tel_passive. With tel active, E_mesh differentiates its own sizing
    /// field, and a field that depends on the prescription would make that coupling
    /// structural rather than accidental -- that bug produced gradients ~1e9 against
    /// E_map's 1.6.
    ///
    /// The field acts ONLY through E_mesh, which w_mesh scales. Remeshing splits when
    /// they improve the TOTAL energy, so at w_mesh = 0.01 a 4x finer tel in a thin band
    /// is a (4)^2 = 16x local E_mesh signal, down-weighted 100x. The global cap worked at
    /// that weight because it changed tel by ~3000x. So expect to need a larger w_mesh
    /// for this to bite, and check n_faces_T_band in the report before believing it did.
    bool map_driven_tel = false;

    /// Target edge length on the seam itself. The one knob that costs: measured on the L,
    /// 0.015 is ~1.55x the faces, 0.0075 is ~3.75x, 0.00375 is ~12.6x. Do not go below
    /// the local INPUT edge length (0.0629 on the L, 0.0149 on the U) expecting more
    /// information -- the prescribed field is constant within an input face, so finer
    /// than that only resolves the seam's position, not its values.
    double map_tel_floor = 0.015;

    /// Gradation: tel grows as floor + g * distance-to-seam. A size-per-length, so it is
    /// independent of input mesh resolution. 0.2-1.0 is the usual meshing range; the cost
    /// is nearly flat in g (3-12% of face count from 0.25 to infinity), so there is no
    /// reason to tune it.
    double map_tel_gradation = 0.5;

    /// Rings the seam is dilated by before the gradation starts. Needed because tel is
    /// read by a single centroid sample per T triangle, so a band thinner than an element
    /// is invisible to it. 2 is the safe default; 1 is marginal on a mesh as fine as the U.
    int map_tel_min_rings = 2;

    /// Jump magnitude above which an edge counts as seam, in log-singular-value units.
    /// Well separated in practice: the L's seam doubles sigma_max, i.e. log 2 = 0.69,
    /// while a flat region reads ~1e-15, and any threshold in [1e-4, 1e-2] picked out the
    /// identical face set when measured.
    double map_tel_jump_thresh = 0.05;

    /// Run the remeshing passes but take NO Newton step: optimize_map returns
    /// immediately, before touching any vertex.
    ///
    /// Diagnostic. Paired with START_FROM_GROUND_TRUTH it holds the map EXACTLY at the
    /// ground truth (remeshing preserves it: splits, collapses and flips were measured
    /// at zero drift) while T is refined to whatever the sizing field asks for. The
    /// reported residual is then not map error at all -- it is purely the discrepancy
    /// between the J* the energy looks up and the J* the true map implies on that
    /// triangle, as a function of T's resolution. corr_mean staying at ~1e-17 is the
    /// built-in proof that the map really did not move.
    ///
    /// Cannot be done by setting max_iterations = 0: the outer loop in
    /// optimize_with_remeshing is `for (iter < max_iterations) { remesh; optimize; }`,
    /// so zero iterations means no remeshing either. Nor by raising
    /// conv_thresh_newton_decrement: that test runs AFTER the line search, so a step is
    /// already taken each round.
    bool remesh_only = false;

    bool probe_symmetry = false;

    double hessian_projection_eps = 1e-9;
    double w_identity = 1e-9;
    int use_cg_above_nnz_per_row = 50;
    double cg_solver_tolerance = 1e-9;

    // Convergence criteria
    double conv_thresh_newton_decrement = 0.01;
    double landmark_threshold = -INF_DOUBLE; // Stop if remeshing converged and landmark distance in world space is under thresh

    // Remeshing
    bool allow_splits = true;
    bool allow_collapses = true;
    bool allow_flips = true;

    void set_adaptive_tel_params(double approx_e, double min_el, double max_el)
    {
        approx_error = approx_e;
        min_edge_length = min_el;
        max_edge_length = max_el;
    }
};

}
