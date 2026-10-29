/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Baseline comparison for the prescribed Jacobian work.
 *
 * Runs 4 configurations over the L-shape meshes:
 *
 *                          | Symmetric Dirichlet | Prescribed metric
 *   -----------------------+---------------------+-------------------
 *   A on A (newL -> newL)  |  consistency check  |  consistency check
 *   A on B (newL -> strL)  |  baseline           |  the real run
 *
 * "A on A" maps a mesh to ITSELF. The correct answer is the identity map, and the
 * optimization must not deviate from it -- a pure consistency check.
 *
 * It is also a sharp test of the new energy: when both meshes are equal we have
 * J* = I everywhere, so the prescribed metric energy reduces EXACTLY to symmetric
 * Dirichlet (tr(M) + tr(M^-1) = ||J||^2 + ||J^-1||^2). The two energy columns must
 * therefore agree on the A-on-A row; if they disagree, the prescribed path has a bug.
 * (For J = I the SVD is degenerate and U, V are arbitrary, but M* = V*I*V^T = I
 * regardless -- the metric form is immune to that ambiguity.)
 *
 * Read BOTH metrics; they can disagree sharply and each catches what the other misses.
 * Ground-truth correspondence error is the primary one, because the residual is what we
 * minimize (a training loss) and a metric prescription has gauge freedom, so it can sit
 * at ~0 while the map is visibly wrong. But the converse happens too: our best
 * from-scratch map has corr 0.0074 and residual 0.51, while a ground-truth start gives
 * corr 0.0132 and residual 0.011 -- worse correspondence, 46x better residual. Those are
 * two different local minima, which is how the basin problem was isolated.
 *
 * Set one flag below, re-run, get another 4 configs x 4 stages of CSV.
 */
#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/FarthestPointSampling.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/Viewer/Colors.hh> // pure Vec4f constants, no OpenGL
#if ISM_HAS_VIEWER
#include <SurfaceMaps/Viewer/MeshView.hh>
#endif
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedJacobianField.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeWithRemeshing.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedResidual.hh>
#if ISM_HAS_VIEWER
#include <SurfaceMaps/AdaptiveTriangulations/Visualization.hh>
#endif
#include <SurfaceMaps/MultiRes/MultiResSphereEmbedding.hh>

#include <Eigen/SVD>
#include <TinyAD/Utils/Timer.hh>
#include <fstream>
#include <cstdlib>

namespace SurfaceMaps
{

// ---------------------------------------------------------------------------
// Settings. Toggle these and re-run to get another set of 4 results.
//
// Two former ablations are gone because their answer is final:
//  - Default-3 landmarks. Pathological: v0=(2,1,0), v1=(1,2,0), v2=(1.33,1.33,0) are
//    coplanar at z=0 and clustered within 1.4 units on a mesh of diameter 4.05, so
//    align_rotation fits a global sphere rotation from nothing. That is what produced
//    the whole-surface rotation plus two bunching singularities. FPS always now.
//  - DISABLE_REMESHING. Refinement is essential to the prescription: with it off T
//    stays at ~15 vertices, the residual sticks at ~0.10 against 0.008, and prescribed
//    degenerates to dirichlet. No reason to run it again.
// ---------------------------------------------------------------------------

/// The mesh pair. Both meshes MUST share connectivity: the ground-truth correspondence
/// metric assumes vertex i of A maps to vertex i of B, and extract_jacobian_field pairs
/// faces by index. Verify with
///   diff <(sed 's#/[0-9]*##g' A.obj | grep '^f ') <(sed 's#/[0-9]*##g' B.obj | grep '^f ')
/// (strip the /vt/vn indices first -- normals legitimately differ between the two).
///
///   l_shape: 786 verts. Identity for x <= 2.08, x -> 2x-1.93 for x >= 2.42, ~101-face
///            transition band. Flat faces + sharp edges, so all curvature sits at the
///            cone points and the developable regions give the energy soft modes.
///   u_shape: 5852 verts. Base stretched 2x in x, both arms translated +-0.65 rigidly,
///            transition at |x| ~ 0.65. 7.4x bigger, so runs are substantially slower.
const std::string MESH_DIR = "meshes/l_shape";
const std::string MESH_A   = "newL.obj";
const std::string MESH_B   = "stretchedL.obj";

/// 8 / 16 / 32 swept. 16 was marginally best (corr 0.0062 vs 0.0074) and gave the widest
/// prescribed-vs-dirichlet gap (4.6x); 32 degrades 2.3x and collapses the gap to 1.03x,
/// because hard landmark constraints start determining the answer instead of the energy.
/// Held at 8 so the W_MESH sweep below is directly comparable to the W_MAP sweep.
const int  N_FPS_LANDMARKS = 16;
const bool OPEN_VIEWER     = true; // false = write screenshots instead

/// Ablation, A-on-B configs only: give mesh B the SAME sphere embedding as mesh A.
/// The two L meshes share connectivity, so the initial map is then exactly the
/// ground-truth correspondence, and the question becomes whether the energy HOLDS it.
/// Our sharpest diagnostic: at W_MAP = 100 it showed dirichlet drifting AWAY from the
/// correct map (0.0160 -> 0.0258) while prescribed holds and improves it
/// (0.00515 -> 0.00332) -- a 7.8x gap, and the demonstration that the correct map is
/// not the as-isometric-as-possible one.
const bool START_FROM_GROUND_TRUTH = false;

/// Run the COARSE phase with the prescribed energy too, not just the fine phase.
/// At W_MAP = 1 this helped from scratch (0.0471 -> 0.0326) and hurt from the
/// ground-truth start (0.0132 -> 0.0151): a noisy-but-roughly-right signal that
/// dominates noise when far from the answer and contributes only noise when already
/// there. Worth re-checking now that the weights have moved, so it stays a flag.
const bool PRESCRIBED_IN_COARSE_PHASE = true;

/// Weight of the map-distortion term -- the only term the prescription lives in.
/// SWEEP DONE (1 / 10 / 100): the prescription was being outvoted. At 100, A-on-B
/// prescribed improves 0.0326 -> 0.00742 corr_mean, the prescribed-vs-dirichlet gap
/// widens 1.4x -> 3.5x, and the A-on-A self-map drift collapses 0.0195 -> 0.000427
/// (which retired the "continuous optimizer breaks the symmetry" mystery -- it was
/// the weighting all along).
/// HOLD AT 1.0 while sweeping W_MESH below: that separation is the point of the sweep.
const double W_MAP = 1.0;

/// Weight of the mesh-quality term. THIS IS THE CURRENT SWEEP: 1 / 0.1 / 0.01, with
/// W_MAP held at 1.
///
/// Why this term specifically. eval_mesh_energy_triangle_T builds an ideal EQUILATERAL
/// triangle of side tel and runs symmetric Dirichlet between the lifted T triangle and
/// that ideal -- once per input mesh. So it demands every T triangle be equilateral on
/// A *and* equilateral on B, and those two demands together imply J is a rotation.
/// E_mesh therefore carries a hidden isometry prior on the map, and it is mutually
/// UNSATISFIABLE with an anisotropic prescription: a triangle cannot be equilateral on
/// both sides when sigma = (2, 1). E_approx, by contrast, is indifferent to J -- it only
/// wants T to hug the surfaces.
///
/// So E_mesh is the likely antagonist, and W_MAP = 100 escaped it only incidentally, by
/// down-weighting everything else at once. Prediction: lowering W_MESH alone recovers
/// most of the W_MAP = 100 gain. If it does, the honest fix is prescription-aware ideal
/// triangles -- apply the already-looked-up V and Sigma to the ideal on the B side --
/// rather than a 100x multiplier on the map term.
const double W_MESH = 1.0;

/// PRIORITY 3: make E_mesh prescription-aware instead of down-weighting it.
///
/// E_mesh builds an ideal EQUILATERAL triangle of side tel and runs symmetric Dirichlet
/// between the lifted T triangle and that ideal, once per input mesh -- so it demands
/// every T triangle be equilateral on A *and* on B, which together imply J is a rotation.
/// That isometry prior is mutually unsatisfiable with an anisotropic prescription.
///
/// With this on, the B-side ideal instead carries the prescribed anisotropy: the ideal's
/// edge vectors are multiplied by S = (M*/sqrt(det M*))^(1/2), which is SPD with det 1,
/// so the ideal gets the prescribed SHAPE and keeps the sizing field's AREA. Satisfying
/// both terms then implies J ~ J* rather than J ~ rotation.
///
/// WHY THIS AND NOT MORE WEIGHT TUNING. On the sphere pair, at matched T resolution
/// (2188 vs 2120 faces) and where T out-resolves the field 6x so the lookup is not
/// binding, dropping W_MESH 1.0 -> 0.1 improved the PRESCRIBED residual 4.4x
/// (0.0687 -> 0.0156) while improving the DIRICHLET control only 1.18x
/// (0.1571 -> 0.1329). That differential is the isometry prior, isolated. And no single
/// W_MESH works across shapes (L wants 0.01, U wants 1.0, sphere wants 0.1), so
/// down-weighting leaves a per-mesh tuning parameter behind.
///
/// PREDICTION, pre-registered against the nine baselines already on disk: with this on at
/// W_MESH = 1, each shape should reach roughly its own best-W_MESH residual --
/// L ~0.051, U ~0.223, sphere ~0.014 -- untuned.
///
/// Reduces EXACTLY to the old ideal when J* = I, so the A-on-A rows are the correctness
/// check: they must stay bit-identical to a run with this off. (Verified: all six A-on-A
/// rows across the three shapes are bit-identical.)
///
/// RESULT (2026-10-22), in two parts -- read both, the first one alone is misleading.
///
/// FROM THE DEFAULT INITIALIZATION the prediction above FAILED:
///   - Alone at W_MESH = 1 it buys 1.21x (L), 1.29x (sphere), and COSTS 14% (U).
///   - W_MESH down-weighting alone buys 4.45x (L), 4.89x (sphere).
///   - On top of W_MESH = 0.01 it buys NOTHING: 0.83x (L), 0.95x (sphere).
/// So the isometry prior is a real but minor effect, fully subsumed by down-weighting.
/// E_mesh's harm is its MAGNITUDE, not its BIAS -- giving it the correct anisotropic
/// target changes little; how hard it pushes is what matters.
///
/// FROM THE MOBIUS-FITTED INITIALIZATION it works, and the first reading was confounded.
/// With MOBIUS_FITTED_EMBEDDING_B = true and W_MESH = 0.01 on the L:
///   mobius alone  0.05113 -> 0.04663 (1.10x)   -- despite a 1.76x better INITIAL residual
///   pmesh alone   0.05113 -> 0.06145 (0.83x)   -- hurts
///   BOTH          0.05113 -> 0.03515 (1.45x)   -- vs 0.91x if the two were independent
/// A real interaction (~1.6x). Mobius supplies a better basin; E_mesh's isometry prior
/// drags the map back off it; pmesh removes the prior so the advantage survives. Neither
/// alone is worth anything.
///
/// BEST KNOWN CONFIGURATION on the L: W_MESH = 0.01 + PRESCRIBED_MESH_IDEAL
///   + MOBIUS_FITTED_EMBEDDING_B. Prescribed 0.03515 res / 0.00247 corr / 0.01918
///   corr_max against a dirichlet 0.10924 / 0.01443 / 0.07048 and an A-on-A floor of
///   0.01576 / 0.00143 / 0.01861 -- i.e. 3.1x / 5.8x / 3.7x over dirichlet, with the
///   WORST-CASE correspondence error now sitting AT the self-map floor.
///
/// Parked OFF so this file still reproduces the plain baseline; flip both (and W_MESH)
/// to get the best result. Its one regression (the U, 0.86x) is lookup-driven: pmesh
/// consumes the centroid lookup's output, so it amplifies lookup error where the lookup
/// is worst. Its benefit is monotone in T's resolution relative to the prescribed field
/// -- U 0.16x of the field: 0.86x; L 0.45x: 1.21x; sphere 6.84x: 1.29x.
const bool PRESCRIBED_MESH_IDEAL = false;

/// Give mesh B the MOBIUS-FITTED sphere embedding produced by the `mobius_probe` app
/// (`output/mobius_probe_<shape>/embedding_B_mobius_fitted.obj`), instead of the one
/// multi_res_sphere_embedding + align_rotation produced.
///
/// The probe fits a 6-DOF Mobius transformation of B's embedding to minimize the INITIAL
/// residual: on the L that takes 0.2303 -> 0.1302 (1.77x), of which a landmark-only fit
/// reaches 0.1466 (84% of the achievable gain). A Mobius transformation of the sphere is a
/// diffeomorphism, so the transformed embedding is still bijective, and J* is untouched
/// (extract_jacobian_field reads only the two meshes).
///
/// Its purpose here is as a CONTROL on the priority-3 refutation: if E_mesh's isometry
/// prior looked minor only because the map starts far from the answer, then starting
/// closer should make PRESCRIBED_MESH_IDEAL matter more. Applies to A-on-B configs only --
/// SelfMap must keep both embeddings identical, and START_FROM_GROUND_TRUTH overrides it.
const bool MOBIUS_FITTED_EMBEDDING_B = false;

/// Treat E_mesh's target edge length as the passive SIZING FIELD it is, instead of
/// re-computing it as a differentiable function of the map. See
/// AdaptiveTriangulationsSettings::mesh_tel_passive for the mechanism; in short, the
/// active re-computation makes tel a map DOF with d/d(tel) ~ -2 l^2/tel^3 against a
/// sizing floor of 1e-3, so E_mesh contributes ~1e9 of gradient where E_map contributes
/// ~1.6, and the strict `<` tie-break routes all of it onto mesh 0.
///
/// It is in the OUTPUT TAG because it is not a tuning knob: every CSV without
/// `_telpassive` in its name was produced with the bug live, and the two are not
/// comparable. On the U self-map at W_MESH = 0.01 it takes corr_mean from 1.37e-2 to
/// 1.60e-8 -- the "self-map floor" that the first column of every error budget was
/// normalized against was this bug, not a resolution limit.
const bool MESH_TEL_PASSIVE = true;

/// Per-iteration trace, via the optimizer's own two callbacks (nothing in the algorithm
/// changes). _callback_for_optim fires once per Newton iteration, inside the loop, after
/// embeddings_T and the local bases are updated (OptimizeMap.cc "after_optimization");
/// _callback_for_remesh fires three times per remeshing pass (Remeshing.cc
/// "after_splits" / "after_collapses" / "after_flips"). Because they are separate, a
/// change can be attributed to a continuous step or to one specific remeshing operation.
///
/// It also records embedding_drift and min_angle_stats, which turn the self-map floor
/// into something directly attributable -- see those two helpers.
///
/// ANSWERED TWICE.
///
/// (1) Refinement sharpens the MEASUREMENT, it does not degrade the map: the residual
///     jumps at after_splits while corr stays flat across those same splits. Confirmed
///     independently by drift, which is resolution-free and does NOT rise at splits.
///
/// (2) 2026-10-25, U self-map, W_MESH 0.01 and 1.0. The floor is made by the CONTINUOUS
///     optimizer, not by remeshing and not by triangle quality:
///                                        W_MESH 1.0      W_MESH 0.01
///       corr increase from Newton steps     95.2%           95.6%
///       drift_max increase from Newton     100.0%          100.0%
///       net drift from after_collapses     +0.04 deg       -0.33 deg  (healing)
///       net drift from after_flips          0.00 deg        0.00 deg  (exactly zero)
///       final mean min-angle               30.98 deg       27.48 deg  (-11%)
///       final mean drift                    0.37 deg        1.63 deg  (4.5x)
///     Quality barely moves while drift grows 4.5x, so the two are uncorrelated and a
///     smoothing operator cannot be the fix. Frames 0-3 are the sharp bit: splits and
///     collapses take T from 30 to 47 to 40 vertices with drift at EXACTLY 0 and corr at
///     6e-17, then the first Newton step alone puts drift at 2.57 deg and corr at
///     0.0018. See embedding_drift for why that step should be impossible.
///
/// OFF by default: it lifts all input vertices every frame and roughly doubles runtime.
/// Run only the two A-on-A configs and skip A-on-B. For the floor diagnostic the
/// self-map is the whole question and A-on-B just doubles the runtime; on the U that
/// matters. (And on a self-map J* = I, so the dirichlet and prescribed rows are the
/// same computation -- one of the two is enough.)
const bool SELF_MAP_ONLY      = false;

/// Log the A<->B antisymmetric part of the gradient and the Newton direction once per
/// iteration (see AdaptiveTriangulationsSettings::probe_symmetry). Pair with
/// SELF_MAP_ONLY -- it is only meaningful on a self-map.
const bool PROBE_SYMMETRY     = false;

/// Mirror of SELF_MAP_ONLY: run only the two A-on-B configs and skip A-on-A.
///
/// For the GT-START ablation the self-map rows are pure overhead -- SelfMap always
/// shares its embedding, so START_FROM_GROUND_TRUTH does not change them and they
/// duplicate the plain run's A-on-A rows exactly. Since the tel fix put them at
/// 1e-11 they carry no information either. On a 52k-vertex input that is half the
/// runtime for nothing.
const bool STRETCHED_ONLY     = false;

/// Multiply the coarse AND fine phases' target approximation error (0.01 / 0.001) by
/// this factor, which is the ONLY clean way to move T's resolution while holding the
/// input mesh, its geometry and the prescribed field J* exactly fixed.
///
/// tel ~ sqrt(6 * approx_error / curvature), so |F(T)| ~ 1/approx_error: 0.25 gives
/// roughly 4x the faces, 4.0 roughly a quarter.
///
/// This exists to test one claim: that the prescribed energy's advantage over symmetric
/// Dirichlet is set by how finely T samples J*, not by the shape. Measured at
/// |F(T)|/|F(input)|, that advantage runs U 0.22 -> 0.99x, L 0.64 -> 9.88x,
/// sphere 5.79 -> 32x, monotone at two weight settings independently. But the three
/// shapes differ in more than that ratio, so the correlation does not establish cause.
/// Scaling T on ONE shape does: refine the U toward the L's ratio and the advantage
/// should appear; coarsen the L toward the U's and it should vanish.
///
/// NOT testable by subdividing the input. A linear subdivision whose displaced mesh is
/// the piecewise-linear interpolant of the original (verified: agrees to 1.2e-6) leaves
/// J* unchanged -- every child triangle inherits its parent's Jacobian -- so it adds
/// redundant samples of the same field rather than a finer field. It also hands every
/// face-interior vertex zero discrete curvature, which corrupts the sizing field and
/// made T COARSER (2606 -> 920 faces), moving the ratio by the wrong mechanism.
const double APPROX_ERROR_SCALE = 1.0;

/// THE GT LOOKUP MEASUREMENT. Remesh T but never take a Newton step
/// (AdaptiveTriangulationsSettings::remesh_only), so paired with
/// START_FROM_GROUND_TRUTH the map stays EXACTLY at the ground truth while T is refined
/// to whatever APPROX_ERROR_SCALE asks for.
///
/// Why that is the measurement we want: the residual metric compares the ACHIEVED
/// Jacobian of each T triangle against the J* the energy looks up for it. When the map
/// is the ground truth the achieved Jacobian IS the true one, so the residual stops
/// being map error and becomes exactly "how wrong is the looked-up J* on this triangle".
/// The lookup is a single point sample -- the input face of A's embedding containing the
/// T triangle's CENTROID -- so this measures the error of representing a field that
/// varies across the triangle by one sample at its middle, plus the error of projecting
/// that face's V frame into the T triangle's own basis.
///
/// corr_mean must stay at ~1e-17 in every stage of such a run. That is the built-in
/// proof that the map did not move and the number means what it claims.
const bool REMESH_ONLY = false;

/// Cap on the sizing field, overriding the phases' default of 100.0 -- i.e. no cap at
/// all, on a mesh normalized to surface area 1.
///
/// The knob APPROX_ERROR_SCALE cannot reach. tel = sqrt(6*e/kappa - 3*e^2) CLAMPED to
/// [min_edge_length, max_edge_length]; on a flat region kappa -> 0 sends tel to infinity
/// and the clamp pins it at max_edge_length whatever e is. So scaling the approximation
/// error refines T only where it is already curved and leaves flat regions exactly as
/// coarse -- and on the U that is precisely where the J* lookup error lives.
///
/// Measured by lookup_distortion_probe at the frozen ground truth, U, approx scale 1:
///   max_el 100 -> |F(T)| 1996, lookup error 0.0473, 17.2% of area off by >10%
///   max_el 0.3 -> |F(T)| 3324, lookup error 0.00446
///   max_el 0.1 -> |F(T)| 3534, lookup error 0.00269
///   max_el 0.03-> |F(T)| 4628, lookup error 0.00120
/// 39x less lookup error for 2.3x the faces. The L barely moves (0.0035 -> 0.0046), so
/// the mechanism is specific to shapes with large flat regions.
const double MAX_EDGE_LENGTH = 100.0;

const bool TRACE_ITERATIONS   = false;
const bool TRACE_WRITE_MESHES = false;
/// Dump a mesh only every Nth traced step (the CSV always gets every step).
const int  TRACE_MESH_STRIDE  = 1;

/// Checkerboard transfer settings.
/// Texcoords are built as (point[(dir+1)%3], point[(dir+2)%3]), so dir = 2 gives
/// (x, y) -- the right choice for the L, which lies in the xy-plane and is extruded
/// along z. (dir = 1 would give (z, x), i.e. constant across the L's main faces,
/// which degenerates the checkerboard into stripes.)
/// Raise TEXTURE_FACTOR if the checkers are too coarse: init_map normalizes the
/// meshes to unit surface area, so their coordinates only span ~0.5.
const int    TEXTURE_PROJECTION_DIR = 2;
const double TEXTURE_FACTOR         = 4.0;

// ---------------------------------------------------------------------------

/// Compact, filesystem-safe tag for a non-default weight:
/// 100 -> "_wmap100", 0.01 -> "_wmesh0p01", and "" at the default of 1.0 so runs at the
/// default keep the output names they already have.
///
/// Anything that changes the RESULT must appear in the output directory name. The sphere
/// embedding is cached per config dir, and a stale one silently turns an ablation into a
/// no-op -- that trap already cost us one plausible-looking but meaningless run.
std::string weight_tag(
        const std::string& _name,
        const double _w)
{
    if (_w == 1.0)
        return "";

    std::string s = std::to_string(_w); // e.g. "0.010000"
    while (!s.empty() && s.back() == '0')
        s.pop_back();
    if (!s.empty() && s.back() == '.')
        s.pop_back();
    std::replace(s.begin(), s.end(), '.', 'p');

    return "_" + _name + s;
}

enum class PairMode
{
    SelfMap,   // "A on A": newL -> newL. Correct answer is the identity map.
    Stretched, // "A on B": newL -> stretchedL. The actual problem.
};

enum class EnergyMode
{
    SymmetricDirichlet, // Original algorithm
    PrescribedMetric,   // Metric-form prescribed Jacobian
};

struct Config
{
    PairMode pair_mode;
    EnergyMode energy_mode;

    std::string name() const
    {
        const std::string pair = (pair_mode == PairMode::SelfMap) ? "AonA" : "AonB";
        const std::string energy = (energy_mode == EnergyMode::SymmetricDirichlet) ? "dirichlet" : "prescribed";
        const std::string lm = "fps" + std::to_string(N_FPS_LANDMARKS);
        // START_FROM_GROUND_TRUTH changes WHICH embedding is used, so it has to be in
        // the name -- see the note on weight_tag above.
        const std::string gt = (pair_mode == PairMode::Stretched && START_FROM_GROUND_TRUTH) ? "_gt" : "";
        return pair + "_" + energy + "_" + lm + gt
             + weight_tag("wmap", W_MAP) + weight_tag("wmesh", W_MESH)
             + (PRESCRIBED_MESH_IDEAL ? "_pmesh" : "")
             + (MESH_TEL_PASSIVE ? "_telpassive" : "")
             + weight_tag("approx", APPROX_ERROR_SCALE)
             + (REMESH_ONLY ? "_frozen" : "")
             + weight_tag("mel", MAX_EDGE_LENGTH / 100.0)
             + ((pair_mode == PairMode::Stretched && MOBIUS_FITTED_EMBEDDING_B
                 && !START_FROM_GROUND_TRUTH) ? "_mob" : "");
    }
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Ground-truth correspondence error.
/// The two L meshes share connectivity, so the true image of vertex v of mesh A is
/// vertex v of mesh B. (In the A-on-A self-map case the two meshes are identical, so
/// this is the identity map and the error must stay at ~0.)
/// Errors are normalized by sqrt(area of B) so they are
/// comparable across meshes (standard in the correspondence literature).
/// This is the primary metric: unlike the metric residual, it is not the quantity
/// being optimized, and it detects material "sliding" that leaves the residual at ~0.
void compute_correspondence_error(
        const MapState& _map_state,
        double& _mean_error,
        double& _median_error,
        double& _max_error)
{
    const TriMesh mapped_A = map_vertices_to_target(_map_state, 0, 1);
    const TriMesh& mesh_B = _map_state.meshes_input[1];
    ISM_ASSERT_EQ(mapped_A.n_vertices(), mesh_B.n_vertices());

    // Normalization: sqrt of B's surface area
    double area_B = 0.0;
    for (auto fh : mesh_B.faces())
        area_B += mesh_B.calc_face_area(fh);
    const double scale = 1.0 / std::sqrt(area_B);

    std::vector<double> errors;
    errors.reserve(mesh_B.n_vertices());
    for (auto vh : mesh_B.vertices())
        errors.push_back((mapped_A.point(vh) - mesh_B.point(vh)).norm() * scale);

    _mean_error = 0.0;
    _max_error = 0.0;
    for (const double e : errors)
    {
        _mean_error += e;
        _max_error = std::max(_max_error, e);
    }
    _mean_error /= (double)errors.size();

    std::sort(errors.begin(), errors.end());
    _median_error = errors[errors.size() / 2];
}

// ResidualStats / weighted_percentile / compute_metric_residual now live in
// AdaptiveTriangulations/PrescribedResidual.hh. They were a private copy here and
// drifted out of sync with the energy they mirror -- see the note in that header.

/// Report both metrics at a pipeline checkpoint, and append a row to the CSV.
/// Measuring at every stage shows exactly WHERE the map drifts off ground truth.
/// Angular distance, in degrees, between the two sphere positions T's vertices carry.
///
/// For the SELF-MAP configs this IS the floor, read straight off the degrees of freedom.
/// Both embeddings start bit-identical, and identical positions mean the identity map at
/// ANY resolution of T -- the map is the PAIR, so a coarse T represents the identity
/// exactly. So every bit of self-map corr_mean and residual is the two embeddings having
/// drifted apart, and nothing else. Measuring that directly removes the lifting, the
/// barycentric interpolation and the per-face SVD from between the cause and the number.
/// (Verified: the trace's first four frames take T from 30 to 47 to 40 vertices through
/// splits and collapses with drift at exactly 0 and corr pinned at 6e-17.)
///
/// AND THE FIRST NEWTON STEP SHOULD NOT BE ABLE TO MOVE IT. On a self-map the two input
/// meshes, their embeddings and their lifting are the same object, so
///     E(x_A, x_B) = q(x_A) + q(x_B) + E_map(x_A, x_B)
/// with ONE q (E_mesh, E_approx, barrier and regularizer each depend on a single
/// embedding) and an E_map that is symmetric under swapping its arguments. A symmetric
/// function has a symmetric gradient and Hessian at a symmetric point, so the Newton
/// step is symmetric and x_A = x_B is preserved exactly. The trace nonetheless goes
/// 0 -> 2.57 deg in one iteration, which leaves two possibilities:
///   (a) something in the assembly treats mesh 0 and mesh 1 differently -- a bug, and
///       the one worth finding, since it would collapse the floor outright; or
///   (b) the symmetric configuration is a SADDLE in the antisymmetric directions, and
///       projected Newton flips those negative eigenvalues positive and takes a large
///       step along them, amplifying a 1e-16 asymmetry to O(1) in a single iteration.
/// Distinguishing them needs the antisymmetric part of the gradient at frame 0:
/// ||g_A - g_B|| / ||g|| at machine epsilon means (b), anything larger means (a).
void embedding_drift(
        const MapState& _map_state,
        double& _mean_deg,
        double& _max_deg)
{
    _mean_deg = 0.0;
    _max_deg = 0.0;
    if (_map_state.embeddings_T.size() < 2 || _map_state.mesh_T.n_vertices() == 0)
        return;

    double sum = 0.0;
    for (auto vh : _map_state.mesh_T.vertices())
    {
        const Vec3d a = _map_state.embeddings_T[0][vh].normalized();
        const Vec3d b = _map_state.embeddings_T[1][vh].normalized();
        const double deg = std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * 180.0 / M_PI;
        sum += deg;
        _max_deg = std::max(_max_deg, deg);
    }
    _mean_deg = sum / (double)_map_state.mesh_T.n_vertices();
}

/// Smallest interior angle per triangle, in degrees: averaged over faces, and the worst
/// face. This is the quantity E_mesh is there to protect (its ideal is equilateral, i.e.
/// 60 deg), so it says whether down-weighting E_mesh actually degrades T's shape or
/// whether the term is being blamed for something else.
void min_angle_stats(
        const TriMesh& _mesh,
        double& _mean_deg,
        double& _worst_deg)
{
    _mean_deg = 0.0;
    _worst_deg = 180.0;
    if (_mesh.n_faces() == 0)
        return;

    double sum = 0.0;
    for (auto fh : _mesh.faces())
    {
        auto heh0 = fh.halfedge();
        auto heh1 = heh0.next();
        auto heh2 = heh1.next();
        const double a = _mesh.calc_sector_angle(heh0);
        const double b = _mesh.calc_sector_angle(heh1);
        const double c = _mesh.calc_sector_angle(heh2);
        const double m = std::min({a, b, c}) * 180.0 / M_PI;
        sum += m;
        _worst_deg = std::min(_worst_deg, m);
    }
    _mean_deg = sum / (double)_mesh.n_faces();
}

void report(
        const MapState& _map_state,
        const std::string& _config_name,
        const std::string& _stage,
        std::ofstream& _csv)
{
    double corr_mean, corr_median, corr_max;
    compute_correspondence_error(_map_state, corr_mean, corr_median, corr_max);

    ResidualStats res;
    compute_metric_residual(_map_state, res);

    double drift_mean, drift_max;
    embedding_drift(_map_state, drift_mean, drift_max);

    const std::vector<TriMesh> lifted_for_quality = lifted_meshes_from_mapstate(_map_state);
    double ang_mean_A, ang_worst_A, ang_mean_B, ang_worst_B;
    min_angle_stats(lifted_for_quality[0], ang_mean_A, ang_worst_A);
    min_angle_stats(lifted_for_quality[1], ang_mean_B, ang_worst_B);

    ISM_INFO("[" << _config_name << " | " << _stage << "] "
             << "corr_err mean=" << corr_mean << " median=" << corr_median << " max=" << corr_max
             << " | residual mean=" << res.mean << " (aw " << res.mean_aw << ")"
             << " median=" << res.median << " p90=" << res.p90 << " max=" << res.max
             << " frac>0.1=" << res.frac_gt_0p1
             << " | area=" << res.area_part_aw << " shape=" << res.shape_part_aw
             << " (" << res.n_faces << " faces)"
             << " | |V(T)|=" << _map_state.mesh_T.n_vertices()
             << " | drift mean=" << drift_mean << " max=" << drift_max << " deg"
             << " | min-angle A " << ang_mean_A << "/" << ang_worst_A
             << " B " << ang_mean_B << "/" << ang_worst_B);

    _csv << _config_name << "," << _stage << ","
         << corr_mean << "," << corr_median << "," << corr_max << ","
         << res.mean << "," << res.mean_aw << ","
         << res.median << "," << res.median_aw << "," << res.p90 << "," << res.max << ","
         << res.frac_gt_0p1 << "," << res.frac_gt_0p1_aw << ","
         << res.area_part_aw << "," << res.shape_part_aw << ","
         << res.n_faces << "," << _map_state.mesh_T.n_vertices() << ","
         << drift_mean << "," << drift_max << ","
         << ang_mean_A << "," << ang_worst_A << ","
         << ang_mean_B << "," << ang_worst_B << "\n";
    _csv.flush();
}

/// Diagnostic ONLY, and only reachable from the environment, so a normal run is
/// untouched: override individual energy weights without a rebuild. Its purpose is to
/// isolate one term for the symmetry probe -- run with a single weight nonzero and
/// g_anti_rel names the term that treats mesh 0 and mesh 1 differently.
void probe_weight_overrides(
        AdaptiveTriangulationsSettings& _settings)
{
    struct { const char* var; double* w; } knobs[] = {
        { "ISM_W_MAP",     &_settings.w_map     },
        { "ISM_W_MESH",    &_settings.w_mesh    },
        { "ISM_W_APPROX",  &_settings.w_approx  },
        { "ISM_W_BARRIER", &_settings.w_barrier },
        { "ISM_W_REG",     &_settings.w_reg     },
    };
    if (const char* e = std::getenv("ISM_TEL_PASSIVE"))
    {
        _settings.mesh_tel_passive = (std::atoi(e) != 0);
        ISM_HIGHLIGHT("[probe] mesh_tel_passive = " << _settings.mesh_tel_passive);
    }
    for (const auto& k : knobs)
    {
        if (const char* e = std::getenv(k.var))
        {
            *k.w = std::atof(e);
            ISM_HIGHLIGHT("[probe] " << k.var << " overridden to " << *k.w);
        }
    }
}

/// Zero-padded frame index, so the dumped meshes sort in optimization order.
std::string pad4(const int _i)
{
    const std::string s = std::to_string(_i);
    return std::string(std::max(0, 4 - (int)s.size()), '0') + s;
}

/// One row of the per-iteration trace, plus (optionally) a mesh pair for the animation.
/// Called from the optimizer's own callbacks, so the state is a completed iteration:
/// embeddings_T are updated, the local bases are re-centered, and the vertex-to-T-face
/// assignments were refreshed inside the line search.
void trace_step(
        const MapState& _map_state,
        const fs::path& _output_dir,
        const std::string& _kind,
        const std::string& _label,
        int& _frame,
        std::ofstream& _csv)
{
    double corr_mean, corr_median, corr_max;
    compute_correspondence_error(_map_state, corr_mean, corr_median, corr_max);

    ResidualStats res;
    compute_metric_residual(_map_state, res);

    double drift_mean, drift_max;
    embedding_drift(_map_state, drift_mean, drift_max);

    const std::vector<TriMesh> lifted = lifted_meshes_from_mapstate(_map_state);
    double ang_mean_A, ang_worst_A, ang_mean_B, ang_worst_B;
    min_angle_stats(lifted[0], ang_mean_A, ang_worst_A);
    min_angle_stats(lifted[1], ang_mean_B, ang_worst_B);

    // Area-weighted only: T changes size constantly here, so the unweighted numbers
    // would not be comparable from one row to the next.
    _csv << _frame << "," << _kind << "," << _label << ","
         << corr_mean << "," << corr_median << ","
         << res.mean_aw << "," << res.area_part_aw << "," << res.shape_part_aw << ","
         << res.frac_gt_0p1_aw << "," << res.n_faces << ","
         << _map_state.mesh_T.n_vertices() << ","
         << drift_mean << "," << drift_max << ","
         << ang_mean_A << "," << ang_worst_A << ","
         << ang_mean_B << "," << ang_worst_B << "\n";
    _csv.flush();

    if (TRACE_WRITE_MESHES && (_frame % std::max(1, TRACE_MESH_STRIDE) == 0))
    {
        const fs::path dir = _output_dir / "trace";
        write_mesh(lifted[0], dir / ("f" + pad4(_frame) + "_" + _kind + "_A.obj"));
        write_mesh(lifted[1], dir / ("f" + pad4(_frame) + "_" + _kind + "_B.obj"));
    }

    ++_frame;
}

// ---------------------------------------------------------------------------
// One configuration
// ---------------------------------------------------------------------------

void run_config(
        const Config& _config,
        const fs::path& _output_root,
        std::ofstream& _csv,
        std::vector<TriMesh>& _out_lifted_A,
        std::vector<TriMesh>& _out_lifted_B,
        std::vector<MapState>& _out_states)
{
    const std::string name = _config.name();
    ISM_INFO("################ " << name << " ################");

    const fs::path output_dir = _output_root / name;
    fs::create_directories(output_dir);

    // A on A maps the mesh to itself: both inputs are the same file.
    const fs::path mesh_path_A = DATA_PATH / MESH_DIR / MESH_A;
    const fs::path mesh_path_B = (_config.pair_mode == PairMode::SelfMap)
            ? mesh_path_A
            : DATA_PATH / MESH_DIR / MESH_B;

    // --- Landmarks -------------------------------------------------------
    // Written into the OUTPUT dir (never into DATA_PATH). The two meshes share
    // connectivity, so identical vertex indices are valid ground-truth pairs.
    const fs::path landmarks_path_A = output_dir / "landmarks_A.pinned";
    const fs::path landmarks_path_B = output_dir / "landmarks_B.pinned";
    {
        const TriMesh mesh_A = read_mesh(mesh_path_A);
        const std::vector<VH> landmarks = farthest_point_sampling(mesh_A, N_FPS_LANDMARKS);
        ISM_INFO("FPS landmarks (" << landmarks.size() << ")");
        write_landmarks_pinned(landmarks, landmarks_path_A, {}, true);
        write_landmarks_pinned(landmarks, landmarks_path_B, {}, true);
    }

    // --- Sphere embeddings ------------------------------------------------
    // The embedding depends on the landmarks, so each config needs its own cache
    // path -- reusing a stale embedding would silently invalidate the comparison.
    const fs::path embedding_path_A = output_dir / "embedding_A.obj";
    const fs::path embedding_path_B = output_dir / "embedding_B.obj";

    // Share one embedding between both meshes when:
    //  - SelfMap: the two inputs ARE the same mesh, so init_map computing two
    //    embeddings independently could yield different ones and the initial map
    //    would not be the identity -- which is exactly what this test must check.
    //  - Stretched + START_FROM_GROUND_TRUTH: the optional ablation.
    const bool share_embedding = (_config.pair_mode == PairMode::SelfMap)
            || START_FROM_GROUND_TRUTH;

    if (share_embedding)
    {
        // Compute mesh A's embedding and use it for BOTH meshes, so the initial map
        // is exactly the ground-truth correspondence (the identity, for SelfMap).
        // init_map only computes an embedding when the file is absent, so writing
        // both files here makes it load ours (and skip its rotation alignment).
        //
        // Cache the embedding ONCE at the output root, then copy it into this config's
        // paths on EVERY run -- never trust the per-config files to have been written
        // with the current settings.
        const fs::path shared_cache = _output_root / "embedding_shared.obj";
        if (!fs::exists(shared_cache))
        {
            TriMesh mesh_A = read_mesh(mesh_path_A);
            center_mesh(mesh_A);
            normalize_surface_area(mesh_A);

            const ExternalProperty<VH, Vec3d> embedding = multi_res_sphere_embedding(mesh_A);
            ISM_ASSERT(sphere_embedding_bijective(mesh_A, embedding));

            write_embedding(mesh_A, embedding, shared_cache);
        }
        fs::copy_file(shared_cache, embedding_path_A, fs::copy_options::overwrite_existing);
        fs::copy_file(shared_cache, embedding_path_B, fs::copy_options::overwrite_existing);
        ISM_INFO("Using shared sphere embedding (initial map = ground truth)");
    }

    // Mobius-fitted embedding for B. Mutually exclusive with share_embedding: that path
    // deliberately makes both embeddings equal, which this would break.
    if (MOBIUS_FITTED_EMBEDDING_B && !share_embedding
        && _config.pair_mode == PairMode::Stretched)
    {
        const fs::path fitted = OUTPUT_PATH
                / ("mobius_probe_" + fs::path(MESH_DIR).filename().string())
                / "embedding_B_mobius_fitted.obj";
        ISM_ASSERT(fs::exists(fitted)); // run `mobius_probe <dir> <A> <B> <fps>` first
        // A's embedding must still be the ORIGINAL one: the fit was computed against it.
        const fs::path src_A = _output_root
                / ("AonB_prescribed_fps" + std::to_string(N_FPS_LANDMARKS)) / "embedding_A.obj";
        ISM_ASSERT(fs::exists(src_A));
        fs::copy_file(src_A, embedding_path_A, fs::copy_options::overwrite_existing);
        fs::copy_file(fitted, embedding_path_B, fs::copy_options::overwrite_existing);
        ISM_INFO("Using the Mobius-fitted embedding for mesh B");
    }

    if (_config.pair_mode == PairMode::SelfMap)
        ISM_INFO("Self-map consistency check: expecting the identity map");

    // --- Init -------------------------------------------------------------
    MapState map_state;
    init_map(map_state,
             { mesh_path_A, mesh_path_B },
             { landmarks_path_A, landmarks_path_B },
             { embedding_path_A, embedding_path_B },
             false);

    // Verify the shared embedding actually took effect. Without this, a stale cached
    // embedding silently turns the ablation into a no-op and the run looks normal.
    if (share_embedding)
    {
        double max_diff = 0.0;
        for (auto vh : map_state.meshes_input[0].vertices())
            max_diff = std::max(max_diff,
                    (map_state.embeddings_input[0][vh] - map_state.embeddings_input[1][vh]).norm());
        ISM_INFO("Shared embedding check: max |emb_A - emb_B| = " << max_diff);
        ISM_ASSERT_L(max_diff, 1e-12);
    }

    map_state.set_distortion_pairs(DistortionPairs::All);

    // Extract the prescribed field for ALL configs: the Dirichlet baselines do not
    // optimize it, but they are still evaluated against it.
    map_state.prescribed_jacobians.push_back(
                extract_jacobian_field(map_state.meshes_input[0], map_state.meshes_input[1]));

    assign_vertices_to_T_faces(map_state);

    report(map_state, name, "00_init", _csv);

    // --- Per-iteration trace ----------------------------------------------
    // tracer(kind) builds a callback for one call site; the optimizer supplies the
    // label ("after_optimization", "after_splits", ...). Everything is captured by
    // reference except the kind, which is copied into the callback.
    std::ofstream trace_csv;
    int trace_frame = 0;
    if (TRACE_ITERATIONS)
    {
        fs::create_directories(output_dir / "trace");
        trace_csv.open(output_dir / "trace.csv");
        trace_csv << "frame,kind,label,corr_mean,corr_median,"
                     "residual_mean_aw,residual_area_aw,residual_shape_aw,"
                     "residual_frac_gt_0p1_aw,n_faces_T,n_verts_T,"
                     "drift_mean_deg,drift_max_deg,"
                     "minang_mean_A,minang_worst_A,minang_mean_B,minang_worst_B\n";
    }
    auto tracer = [&] (const std::string& _kind)
    {
        return [&, _kind] (const std::string& _label)
        {
            if (TRACE_ITERATIONS)
                trace_step(map_state, output_dir, _kind, _label, trace_frame, trace_csv);
        };
    };

    // --- Pipeline (identical across configs; only the energy differs) ------
    TinyAD::Timer timer(name);

    landmark_phase(map_state);
    report(map_state, name, "01_landmark", _csv);

    const bool prescribed = (_config.energy_mode == EnergyMode::PrescribedMetric);

    // Inlined coarse_phase(): that helper builds its own settings and accepts no
    // overrides, so we replicate its two lines here to be able to change them.
    {
        ISM_INFO("#################### Starting Coarse Phase ####################");
        AdaptiveTriangulationsSettings settings = coarse_phase_settings();
        settings.use_prescribed_jacobian = prescribed && PRESCRIBED_IN_COARSE_PHASE;
        settings.prescribed_metric_form = true;
        settings.prescribed_mesh_ideal = PRESCRIBED_MESH_IDEAL;
        settings.mesh_tel_passive = MESH_TEL_PASSIVE;
        settings.approx_error *= APPROX_ERROR_SCALE;
        settings.remesh_only = REMESH_ONLY;
        settings.max_edge_length = std::min(settings.max_edge_length, MAX_EDGE_LENGTH);
        settings.probe_symmetry = PROBE_SYMMETRY;
        settings.w_map = W_MAP;
        settings.w_mesh = W_MESH;
        probe_weight_overrides(settings);
        optimize_with_remeshing(map_state, settings, "",
                                tracer("coarse_optim"), tracer("coarse_remesh"));
    }
    report(map_state, name, "02_coarse", _csv);

    {
        AdaptiveTriangulationsSettings settings = fine_phase_settings();
        settings.use_prescribed_jacobian = prescribed;
        settings.prescribed_metric_form = true; // metric form, not the legacy full-J* residual
        settings.prescribed_mesh_ideal = PRESCRIBED_MESH_IDEAL;
        settings.mesh_tel_passive = MESH_TEL_PASSIVE;
        settings.approx_error *= APPROX_ERROR_SCALE;
        settings.remesh_only = REMESH_ONLY;
        settings.max_edge_length = std::min(settings.max_edge_length, MAX_EDGE_LENGTH);
        settings.probe_symmetry = PROBE_SYMMETRY;
        settings.w_map = W_MAP;
        settings.w_mesh = W_MESH;
        settings.max_iterations = 50;
        probe_weight_overrides(settings);
        optimize_with_remeshing(map_state, settings, "",
                                tracer("fine_optim"), tracer("fine_remesh"));
    }
    report(map_state, name, "03_final", _csv);

    timer.stop();
    ISM_INFO("[" << name << "] runtime " << timer.seconds() << " s");

    // --- Output -----------------------------------------------------------
    std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(map_state);
    for (int i = 0; i < (int)map_state.meshes_input.size(); ++i)
        write_mesh(lifted_Ts[i], output_dir / ("T_lifted_" + std::to_string(i) + ".obj"));

    _out_lifted_A.push_back(lifted_Ts[0]);
    _out_lifted_B.push_back(lifted_Ts[1]);
    _out_states.push_back(map_state);
}

// ---------------------------------------------------------------------------

void run()
{
    // The mesh pair MUST be part of the output path. Sphere embeddings are cached under
    // output_root (embedding_shared.obj, and embedding_A/B.obj per config) and loaded
    // whenever present, so a shared root would silently feed one pair's embedding to the
    // other -- the same stale-cache trap that once turned START_FROM_GROUND_TRUTH into a
    // no-op and produced a plausible-looking but meaningless run.
    const fs::path output_root =
            OUTPUT_PATH / ("baseline_comparison_" + fs::path(MESH_DIR).filename().string());
    fs::create_directories(output_root);

#if ISM_HAS_VIEWER
    const fs::path screenshot_dir = output_root / "screenshots";
    fs::create_directories(screenshot_dir);
    glow::SharedTexture2D texture = read_texture(DATA_PATH / "textures/checkerboard.png");
#endif

    // CSV of all metrics, one row per (config, stage)
    const std::string suffix = "fps" + std::to_string(N_FPS_LANDMARKS)
            + std::string(START_FROM_GROUND_TRUTH ? "_gt" : "")
            + std::string(PRESCRIBED_IN_COARSE_PHASE ? "_pcoarse" : "")
            + weight_tag("wmap", W_MAP) + weight_tag("wmesh", W_MESH)
            + std::string(PRESCRIBED_MESH_IDEAL ? "_pmesh" : "")
            + std::string(MESH_TEL_PASSIVE ? "_telpassive" : "")
            + weight_tag("approx", APPROX_ERROR_SCALE)
            + std::string(REMESH_ONLY ? "_frozen" : "")
            + weight_tag("mel", MAX_EDGE_LENGTH / 100.0)
            + std::string(MOBIUS_FITTED_EMBEDDING_B ? "_mob" : "");
    std::ofstream csv(output_root / ("metrics_" + suffix + ".csv"));
    csv << "config,stage,corr_mean,corr_median,corr_max,"
           "residual_mean,residual_mean_aw,residual_median,residual_median_aw,"
           "residual_p90,residual_max,residual_frac_gt_0p1,residual_frac_gt_0p1_aw,"
           "residual_area_aw,residual_shape_aw,"
           "n_faces_T,n_verts_T,"
           "drift_mean_deg,drift_max_deg,"
           "minang_mean_A,minang_worst_A,minang_mean_B,minang_worst_B\n";

    ISM_ASSERT(!(SELF_MAP_ONLY && STRETCHED_ONLY)); // they exclude each other
    const std::vector<Config> configs = SELF_MAP_ONLY
        ? std::vector<Config>{
            { PairMode::SelfMap,   EnergyMode::SymmetricDirichlet },
          }
        : STRETCHED_ONLY
        ? std::vector<Config>{
            { PairMode::Stretched, EnergyMode::SymmetricDirichlet },
            { PairMode::Stretched, EnergyMode::PrescribedMetric   },
          }
        : std::vector<Config>{
            { PairMode::SelfMap,   EnergyMode::SymmetricDirichlet },
            { PairMode::SelfMap,   EnergyMode::PrescribedMetric   },
            { PairMode::Stretched, EnergyMode::SymmetricDirichlet },
            { PairMode::Stretched, EnergyMode::PrescribedMetric   },
          };

    std::vector<TriMesh> lifted_As, lifted_Bs;
    std::vector<MapState> states;
    for (const auto& config : configs)
        run_config(config, output_root, csv, lifted_As, lifted_Bs, states);

    csv.close();
    ISM_INFO("Wrote metrics_" << suffix << ".csv");

#if !ISM_HAS_VIEWER
    // Headless build (e.g. Apple Silicon): macOS caps desktop OpenGL at 4.1 and glow
    // needs 4.2-4.5, so there is no viewer here. The CSVs, the per-config trace and the
    // lifted-T OBJs are all written exactly as above; only the 8 renders are missing.
    // To inspect a per-face field without a window, colour it with linear_colors() and
    // write it with write_colored_ply_flat().
    ISM_INFO("Headless build: skipping the 8 renders. CSV, trace and OBJs are written.");
#else
    // --- Visualization: 2 per config = 8 total ---------------------------
    // Row 1: geometry + wireframe on B (does T look sane, do triangles slide?)
    // Row 2: checkerboard transferred A -> B (sliding and shear are obvious here)
    for (int i = 0; i < (int)configs.size(); ++i)
    {
        const std::string name = configs[i].name();

        {
            auto s = screenshot_config(OPEN_VIEWER, screenshot_dir / (name + "_wire.png"), tg::ivec2(1920, 1080), true);
            auto style = default_style();
            auto g = gv::grid();
            {
                auto v = gv::view();
                view_mesh(states[i].meshes_input[0], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_As[i], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_As[i], MAGENTA, WidthScreen(0.5));
            }
            {
                auto v = gv::view();
                view_mesh(states[i].meshes_input[1], Color(0.8, 0.8, 0.8, 0.5));
                view_mesh(lifted_Bs[i], Color(1.0, 1.0, 1.0, 0.8));
                view_wireframe(lifted_Bs[i], TEAL, WidthScreen(0.5));
            }
        }

        {
            auto s = screenshot_config(OPEN_VIEWER, screenshot_dir / (name + "_texture.png"), tg::ivec2(1920, 1080), true);
            auto style = default_style();
            auto g = gv::grid();
            {
                auto v = gv::view();
                view_texture_frontal_projection_input(states[i], 0, 0, TEXTURE_PROJECTION_DIR, TEXTURE_FACTOR, texture);
            }
            {
                auto v = gv::view();
                view_texture_frontal_projection_input(states[i], 0, 1, TEXTURE_PROJECTION_DIR, TEXTURE_FACTOR, texture);
            }
        }
    }

#endif // ISM_HAS_VIEWER

    ISM_INFO("Done.");
}

}  // namespace SurfaceMaps

int main()
{
#if ISM_HAS_VIEWER
    glow::glfw::GlfwContext ctx;
#endif
    using namespace SurfaceMaps;
    init_lib_surface_maps();
    run();
    return 0;
}
