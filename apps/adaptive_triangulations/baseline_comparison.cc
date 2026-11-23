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
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeVertex.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AssignVerticesToFaces.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/AdaptiveTargetEdgeLength.hh>
#include <SurfaceMaps/AdaptiveTriangulations/PrescribedResidual.hh>
#include <SurfaceMaps/Misc/ConstantCurvatureEmbedding.hh>
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

/// Mesh pair. Both meshes MUST share connectivity (corr metric and J* pair by index). See docs/baseline_comparison_notes.md#mesh_dir
/// Only read when ALL_MODELS is off; tools/sweep.py patches these three.
const std::string MESH_DIR = "meshes/l_shape";
const std::string MESH_A   = "newL.obj";
const std::string MESH_B   = "stretchedL.obj";

/// One mesh pair to run. See docs/baseline_comparison_notes.md#all_models-models
struct Model
{
    std::string name;   ///< short label, used in the log and the captions only
    std::string dir;    ///< relative to DATA_PATH
    std::string mesh_a;
    std::string mesh_b;
};

/// Run every entry of MODELS back to back instead of just MESH_DIR/MESH_A/MESH_B.
/// See docs/baseline_comparison_notes.md#all_models-models
///
/// Every model finishes, and every CSV is closed, BEFORE the first viewer window opens.
/// The windows block, so rendering each model as it finishes would mean quitting at
/// window 3 loses the later models' numbers entirely; this way they are already on disk.
///
/// tools/sweep.py patches this to false -- it drives one pair per build and relies on
/// MESH_DIR/MESH_A/MESH_B.
const bool ALL_MODELS = true;

/// The four pairs that share connectivity and therefore have a valid J* and corr_err.
/// Mirrors tools/sweep.py's MODELS so the two agree on what "the four models" means.
/// The hand pair is NOT here: data/meshes/hand has 6400 vs 12184 faces, so it needs the
/// J* input path before it can run at all.
const std::vector<Model> MODELS = {
    { "betterL", "meshes/l_shape", "betterL.obj", "betterLstretched.obj" },
    { "L",       "meshes/l_shape", "newL.obj",    "stretchedL.obj"       },
    { "U",       "meshes/u_shape", "u1.obj",      "u1stretch.obj"        },
    { "sphere",  "meshes/sphere",  "sphere.obj",  "ellipse1.obj"         },
};

/// 16 is best measured; 32 over-constrains and collapses the prescribed gap to 1.03x. See docs/baseline_comparison_notes.md#n_fps_landmarks
const int  N_FPS_LANDMARKS = 16;
const bool OPEN_VIEWER     = true; // false = write screenshots instead

/// Release the pins after the landmark phase, as every upstream app does. See docs/baseline_comparison_notes.md#release_landmarks
const bool RELEASE_LANDMARKS = true;

/// Give B mesh A's own embedding, so the initial map is exactly correct. See docs/baseline_comparison_notes.md#start_from_ground_truth
const bool START_FROM_GROUND_TRUTH = false;

/// Use the prescribed energy in the coarse phase too, not only the fine one. See docs/baseline_comparison_notes.md#prescribed_in_coarse_phase
const bool PRESCRIBED_IN_COARSE_PHASE = true;

/// Weight on E_map. Swept; 1.0 is the measured optimum. See docs/baseline_comparison_notes.md#w_map
const double W_MAP = 1.0;

/// Weight on E_mesh. 0.01 is the recommended value; 1.0 is the upstream default. See docs/baseline_comparison_notes.md#w_mesh
const double W_MESH = 0.01;

/// Treat E_mesh's tel as a passive sizing field. Fixes a ~1e9 spurious gradient. See docs/baseline_comparison_notes.md#mesh_tel_passive
const bool MESH_TEL_PASSIVE = true;

/// Run only the A-on-A configs (the self-map floor diagnostic). See docs/baseline_comparison_notes.md#self_map_only
const bool SELF_MAP_ONLY      = false;

/// Run only the A-on-B configs, skipping the self-map pair. See docs/baseline_comparison_notes.md#stretched_only
const bool STRETCHED_ONLY     = true;

/// Drop the symmetric-Dirichlet arm and run only the prescribed energy. The baseline is
/// settled, so a sweep that only varies model and resolution does not need to re-measure
/// it; halves the runtime.
const bool PRESCRIBED_ONLY    = true;

/// Multiplier on both phases' approx_error. 1.0 is the paper's setting. See docs/baseline_comparison_notes.md#approx_error_scale
const double APPROX_ERROR_SCALE = 1.0;

/// Global cap on tel. 0.03 is the recommended value; 100.0 is effectively uncapped. See docs/baseline_comparison_notes.md#max_edge_length
const double MAX_EDGE_LENGTH = 0.03;

const bool TRACE_ITERATIONS   = false;

/// Also dump the lifted T per step as OBJ frames; render with tools/render_trace.py. See docs/baseline_comparison_notes.md#trace_write_meshes
const bool TRACE_WRITE_MESHES = false;
/// Dump a mesh only every Nth traced step (the CSV always gets every step).
const int  TRACE_MESH_STRIDE  = 1;

/// Checkerboard transfer settings. PROJECTION_DIR applies to TexSource::Planar only. See docs/baseline_comparison_notes.md#texture_projection_dir
const int    TEXTURE_PROJECTION_DIR = 2;
const double TEXTURE_FACTOR         = 4.0;

/// Where the checkerboard UVs come from. Read distortion with Triplanar; check what
/// the optimizer sees with Sphere. Planar degenerates on 53.8% of the L's area.
/// See docs/baseline_comparison_notes.md#texture_source
enum class TexSource { Planar, Sphere, Triplanar };
const TexSource TEXTURE_SOURCE = TexSource::Triplanar;

/// Draw landmark balls into the wire and texture renders, not as a render of their own. See docs/baseline_comparison_notes.md#landmark_balls
const bool   LANDMARK_BALLS         = true;
const double LANDMARK_RADIUS_FACTOR = 0.015;

/// Also refine T where the PRESCRIBED metric jumps, not only where the surface curves. See docs/baseline_comparison_notes.md#map_driven_tel
const bool   MAP_DRIVEN_TEL    = false;
const double MAP_TEL_FLOOR     = 0.015;

/// Heat diffusion length w on mesh A (t = w^2). 0 = auto, max(2*FLOOR, mean edge). See docs/baseline_comparison_notes.md#map_tel_diffusion_length
const double MAP_TEL_DIFFUSION_LENGTH = 0.0;

/// Write the three sizing-field PLYs per config; render with tools/render_field.py. See docs/baseline_comparison_notes.md#map_tel_debug_ply
const bool   MAP_TEL_DEBUG_PLY = false;


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
             + (MESH_TEL_PASSIVE ? "_telpassive" : "")
             + weight_tag("approx", APPROX_ERROR_SCALE)
             + weight_tag("mel", MAX_EDGE_LENGTH / 100.0)
             + (MAP_DRIVEN_TEL ? ("_mdtel" + weight_tag("floor", MAP_TEL_FLOOR)) : "")
             + (RELEASE_LANDMARKS ? "_rellm" : "")
             ;
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

/// Flipped-or-degenerate T faces in each of T's two sphere embeddings.
///
/// This IS the algorithm's bijectivity invariant, read straight off the degrees of
/// freedom. The map is "drop each T vertex onto mesh i through sphere i", so it is a
/// bijection exactly when both embeddings_T[i] are valid (non-inverted) spherical
/// triangulations; the barrier term exists to keep them that way. Nonzero here means the
/// map has genuinely lost injectivity, and any rendering artifact is then a symptom.
/// Zero here means a texture artifact is in the VISUALIZATION, not in the map -- which
/// matters because a frontal-projection checkerboard has its own degeneracies (see
/// TEXTURE_PROJECTION_DIR).
void flipped_T_faces(
        const MapState& _map_state,
        int& _flipped_A,
        int& _flipped_B)
{
    _flipped_A = count_flipped(_map_state.mesh_T, _map_state.embeddings_T[0], Spherical);
    _flipped_B = (_map_state.embeddings_T.size() > 1)
            ? count_flipped(_map_state.mesh_T, _map_state.embeddings_T[1], Spherical)
            : 0;
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

/// T faces sitting in the map-driven refinement band, i.e. whose SAMPLED target edge
/// length is close to the floor.
///
/// This is the diagnostic that decides whether MAP_DRIVEN_TEL did anything at all. The
/// field acts only through E_mesh, and remeshing splits only when they improve the TOTAL
/// energy, so at a small W_MESH the band can be asked for and simply never materialize.
/// If this count does not rise when the field is switched on, the field was OUTVOTED, not
/// wrong -- raise W_MESH rather than lowering the floor.
///
/// Sampled exactly the way E_mesh samples it (one lookup at the T face's sphere
/// centroid), so it reports what the optimizer actually saw, not what the field holds.
///
/// Counted whether or not MAP_DRIVEN_TEL is on, because the number is only meaningful as
/// a DIFFERENCE: the curvature field alone already drives some faces below this threshold
/// (85 of 1868 on the L's self-map, where the prescribed field is the identity and the
/// map-driven term is correctly a no-op), so the absolute count is not the band size.
int count_band_T_faces(
        const MapState& _map_state,
        const double _floor)
{
    if (_map_state.tels_input.empty())
        return 0;

    int n = 0;
    for (auto fh : _map_state.mesh_T.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(_map_state.mesh_T, fh, vh_a, vh_b, vh_c);
        const double tel = ideal_triangle_edge_length(
                    _map_state.embeddings_T[0][vh_a],
                    _map_state.embeddings_T[0][vh_b],
                    _map_state.embeddings_T[0][vh_c],
                    _map_state.meshes_input[0], _map_state.meshes_embeddings_input[0],
                    _map_state.tels_input[0], _map_state.bsp_embeddings_input[0]);
        if (tel < 1.5 * _floor)
            ++n;
    }
    return n;
}

/// _report_settings is a FIXED yardstick, not the live objective: every stage is scored
/// with the same weights, so the energies are comparable down the column and across
/// models. The optimizer's own objective changes between phases (w_barrier 1e-3 -> 1e-6,
/// and approx_error), so its per-phase value is not comparable even with itself.
void report(
        const MapState& _map_state,
        const std::string& _config_name,
        const std::string& _stage,
        const AdaptiveTriangulationsSettings& _report_settings,
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

    int flipped_A, flipped_B;
    flipped_T_faces(_map_state, flipped_A, flipped_B);

    const int n_band_T = count_band_T_faces(_map_state, MAP_TEL_FLOOR);

    if (flipped_A != 0 || flipped_B != 0)
    {
        ISM_WARNING("[" << _config_name << " | " << _stage << "] NOT BIJECTIVE: "
                    << flipped_A << " flipped T faces on A, " << flipped_B << " on B, of "
                    << _map_state.mesh_T.n_faces());
    }
    else
    {
        ISM_INFO("[" << _config_name << " | " << _stage << "] bijective: 0 flipped T faces");
    }

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
             << " B " << ang_mean_B << "/" << ang_worst_B
             << (MAP_DRIVEN_TEL ? (" | band faces " + std::to_string(n_band_T)) : ""));

    const double E = total_energy(_map_state, _report_settings);
    const double nF_T = (double)_map_state.mesh_T.n_faces();

    _csv << _config_name << "," << _stage << ","
         << corr_mean << "," << corr_median << "," << corr_max << ","
         << res.mean << "," << res.mean_aw << ","
         << res.median << "," << res.median_aw << "," << res.p90 << "," << res.max << ","
         << res.frac_gt_0p1 << "," << res.frac_gt_0p1_aw << ","
         << res.area_part_aw << "," << res.shape_part_aw << ","
         << E << "," << (nF_T > 0 ? E / nF_T : 0.0) << ","
         << res.n_faces << "," << _map_state.mesh_T.n_vertices() << ","
         << n_band_T << ","
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
        const AdaptiveTriangulationsSettings& _report_settings,
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

    int flipped_A, flipped_B;
    flipped_T_faces(_map_state, flipped_A, flipped_B);
    if (flipped_A != 0 || flipped_B != 0)
    {
        ISM_WARNING("[trace " << _kind << " " << _label << "] NOT BIJECTIVE: "
                    << flipped_A << " flipped T faces on A, " << flipped_B << " on B, of "
                    << _map_state.mesh_T.n_faces());
    }

    // Area-weighted only: T changes size constantly here, so the unweighted numbers
    // would not be comparable from one row to the next.
    const double E = total_energy(_map_state, _report_settings);
    const double nF_T = (double)_map_state.mesh_T.n_faces();

    _csv << _frame << "," << _kind << "," << _label << ","
         << corr_mean << "," << corr_median << ","
         << res.mean_aw << "," << res.area_part_aw << "," << res.shape_part_aw << ","
         << res.frac_gt_0p1_aw << "," << E << "," << (nF_T > 0 ? E / nF_T : 0.0) << ","
         << res.n_faces << ","
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
        const Model& _model,
        const Config& _config,
        const fs::path& _output_root,
        std::ofstream& _csv,
        std::vector<TriMesh>& _out_lifted_A,
        std::vector<TriMesh>& _out_lifted_B,
        std::vector<MapState>& _out_states)
{
    const std::string name = _config.name();
    ISM_INFO("################ " << _model.name << " / " << name << " ################");

    const fs::path output_dir = _output_root / name;
    fs::create_directories(output_dir);

    // A on A maps the mesh to itself: both inputs are the same file.
    const fs::path mesh_path_A = DATA_PATH / _model.dir / _model.mesh_a;
    const fs::path mesh_path_B = (_config.pair_mode == PairMode::SelfMap)
            ? mesh_path_A
            : DATA_PATH / _model.dir / _model.mesh_b;

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

    // The fixed yardstick every energy in this run is scored with -- see report(). Built
    // once, from the fine phase's weights, because that is what the final state was
    // actually optimized under. w_map/w_mesh and the prescribed flags must match the run
    // or the number would not describe this configuration at all.
    AdaptiveTriangulationsSettings report_settings = fine_phase_settings(
                default_fine_approx_error * APPROX_ERROR_SCALE, default_w_approx);
    report_settings.use_prescribed_jacobian = (_config.energy_mode == EnergyMode::PrescribedMetric);
    report_settings.mesh_tel_passive = MESH_TEL_PASSIVE;
    report_settings.max_edge_length = std::min(report_settings.max_edge_length, MAX_EDGE_LENGTH);
    report_settings.w_map = W_MAP;
    report_settings.w_mesh = W_MESH;
    report_settings.hard_landmark_constraints = false; // landmark terms are excluded anyway

    // Build the sizing field up front, so 00_init has a defined energy too. Every phase
    // calls init_target_edge_lengths itself and overwrites this, so it cannot leak into
    // the run -- it only fills in the one row that would otherwise be NaN. With
    // START_FROM_GROUND_TRUTH that row is the map at J == J*, i.e. the energy floor.
    init_target_edge_lengths(map_state, report_settings);
    if (map_state.vertex_areas_input.size() != map_state.meshes_input.size())
        for (const auto& m : map_state.meshes_input)
            map_state.vertex_areas_input.push_back(rel_vertex_areas(m));

    report(map_state, name, "00_init", report_settings, _csv);

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
                     "residual_frac_gt_0p1_aw,energy,energy_per_face,n_faces_T,n_verts_T,"
                     "drift_mean_deg,drift_max_deg,"
                     "minang_mean_A,minang_worst_A,minang_mean_B,minang_worst_B\n";
    }
    auto tracer = [&] (const std::string& _kind)
    {
        return [&, _kind] (const std::string& _label)
        {
            if (TRACE_ITERATIONS)
                trace_step(map_state, output_dir, _kind, _label, report_settings,
                           trace_frame, trace_csv);
        };
    };

    // --- Pipeline (identical across configs; only the energy differs) ------
    TinyAD::Timer timer(name);

    landmark_phase(map_state);

    // Snapshot for the render only. release_landmarks() clears landmarks_T AND every
    // landmarks_input[i], so with RELEASE_LANDMARKS on the final state has nothing left
    // to draw and the landmark overlay would come out empty. Input vertex handles stay
    // valid for the whole run (the input meshes are never remeshed), so these can be put
    // back at the end. landmarks_T CANNOT be: those index mesh_T, which remeshing
    // rewrites, and once released the landmark vertices lose their collapse protection
    // (Remeshing.cc:298), so a stale handle can name a different vertex or none.
    const std::vector<std::vector<VH>> landmarks_input_at_init = map_state.landmarks_input;

    // See RELEASE_LANDMARKS: off is this app's historical behaviour, on matches
    // every upstream app.
    if (RELEASE_LANDMARKS)
        release_landmarks(map_state);
    report(map_state, name, "01_landmark", report_settings, _csv);

    const bool prescribed = (_config.energy_mode == EnergyMode::PrescribedMetric);

    // Inlined coarse_phase(): that helper builds its own settings and accepts no
    // overrides, so we replicate its two lines here to be able to change them.
    {
        ISM_INFO("#################### Starting Coarse Phase ####################");
        AdaptiveTriangulationsSettings settings = coarse_phase_settings();
        settings.use_prescribed_jacobian = prescribed && PRESCRIBED_IN_COARSE_PHASE;
        settings.prescribed_metric_form = true;
        settings.mesh_tel_passive = MESH_TEL_PASSIVE;
        settings.map_driven_tel = MAP_DRIVEN_TEL;
        settings.map_tel_floor = MAP_TEL_FLOOR;
        settings.map_tel_diffusion_length = MAP_TEL_DIFFUSION_LENGTH;
        if (MAP_TEL_DEBUG_PLY)
            settings.map_tel_debug_dir = output_dir / "sizing_field";
        settings.approx_error *= APPROX_ERROR_SCALE;
        settings.max_edge_length = std::min(settings.max_edge_length, MAX_EDGE_LENGTH);
        settings.w_map = W_MAP;
        settings.w_mesh = W_MESH;
        probe_weight_overrides(settings);
        optimize_with_remeshing(map_state, settings, "",
                                tracer("coarse_optim"), tracer("coarse_remesh"));
    }
    report(map_state, name, "02_coarse", report_settings, _csv);

    {
        AdaptiveTriangulationsSettings settings = fine_phase_settings();
        settings.use_prescribed_jacobian = prescribed;
        settings.prescribed_metric_form = true; // metric form, not the legacy full-J* residual
        settings.mesh_tel_passive = MESH_TEL_PASSIVE;
        settings.map_driven_tel = MAP_DRIVEN_TEL;
        settings.map_tel_floor = MAP_TEL_FLOOR;
        settings.map_tel_diffusion_length = MAP_TEL_DIFFUSION_LENGTH;
        if (MAP_TEL_DEBUG_PLY)
            settings.map_tel_debug_dir = output_dir / "sizing_field";
        settings.approx_error *= APPROX_ERROR_SCALE;
        settings.max_edge_length = std::min(settings.max_edge_length, MAX_EDGE_LENGTH);
        settings.w_map = W_MAP;
        settings.w_mesh = W_MESH;
        settings.max_iterations = 50;
        probe_weight_overrides(settings);
        optimize_with_remeshing(map_state, settings, "",
                                tracer("fine_optim"), tracer("fine_remesh"));
    }
    report(map_state, name, "03_final", report_settings, _csv);

    timer.stop();
    ISM_INFO("[" << name << "] runtime " << timer.seconds() << " s");

    // --- Output -----------------------------------------------------------
    std::vector<TriMesh> lifted_Ts = lifted_meshes_from_mapstate(map_state);
    for (int i = 0; i < (int)map_state.meshes_input.size(); ++i)
        write_mesh(lifted_Ts[i], output_dir / ("T_lifted_" + std::to_string(i) + ".obj"));

    // Put the landmark positions back for the overlay (see landmarks_input_at_init).
    // After every report() and every write, so nothing that was measured is touched.
    map_state.landmarks_input = landmarks_input_at_init;

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
    //
    // The MESH FILE NAMES, not just the directory: l_shape holds two different pairs
    // (newL/stretchedL and betterL/betterLstretched), and keying on the directory alone
    // would hand the second pair the first one's cached embedding -- precisely the trap
    // the paragraph above describes, left open because there used to be one pair per
    // directory.
    // So output_root is built per model, inside the loop below.

    // With ALL_MODELS off this is the single MESH_DIR/MESH_A/MESH_B pair, so one code
    // path serves both and the sweep driver only has to flip the flag.
    const std::vector<Model> models = ALL_MODELS
            ? MODELS
            : std::vector<Model>{ { fs::path(MESH_A).stem().string(), MESH_DIR, MESH_A, MESH_B } };

#if ISM_HAS_VIEWER
    glow::SharedTexture2D texture = read_texture(DATA_PATH / "textures/checkerboard.png");
#endif

    // CSV of all metrics, one row per (config, stage). One file per model, under that
    // model's own output_root; tools/collect.py aggregates them into sweep_summary.csv.
    const std::string suffix = "fps" + std::to_string(N_FPS_LANDMARKS)
            + std::string(START_FROM_GROUND_TRUTH ? "_gt" : "")
            + std::string(PRESCRIBED_IN_COARSE_PHASE ? "_pcoarse" : "")
            + weight_tag("wmap", W_MAP) + weight_tag("wmesh", W_MESH)
            + std::string(MESH_TEL_PASSIVE ? "_telpassive" : "")
            + weight_tag("approx", APPROX_ERROR_SCALE)
            + weight_tag("mel", MAX_EDGE_LENGTH / 100.0)
            + (MAP_DRIVEN_TEL ? ("_mdtel" + weight_tag("floor", MAP_TEL_FLOOR)) : "")
            + std::string(RELEASE_LANDMARKS ? "_rellm" : "")
            ;
    ISM_ASSERT(!(SELF_MAP_ONLY && STRETCHED_ONLY)); // they exclude each other
    std::vector<Config> configs = SELF_MAP_ONLY
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
    if (PRESCRIBED_ONLY)
    {
        std::vector<Config> keep;
        for (const auto& c : configs)
            if (c.energy_mode == EnergyMode::PrescribedMetric)
                keep.push_back(c);
        ISM_ASSERT(!keep.empty()); // SELF_MAP_ONLY is dirichlet-only, so they conflict
        configs = keep;
    }

    // Everything one model needs for its three renders, kept alive until the whole
    // sweep has finished -- see ALL_MODELS for why the rendering waits for that.
    struct ModelRun
    {
        Model model;
        fs::path output_root;
        fs::path screenshot_dir;
        std::vector<TriMesh> lifted_As;
        std::vector<TriMesh> lifted_Bs;
        std::vector<MapState> states;
    };
    std::vector<ModelRun> runs;
    // MapState declares a copy constructor, which suppresses the implicit move one, so
    // a reallocation here would deep-copy every state. Reserve and it never happens.
    runs.reserve(models.size());

    for (const Model& model : models)
    {
        ISM_INFO("======== model " << model.name << " ("
                 << model.mesh_a << " -> " << model.mesh_b << ") ========");

        runs.push_back(ModelRun{});
        ModelRun& rec = runs.back();
        rec.model = model;
        rec.output_root = OUTPUT_PATH
                / ("baseline_comparison_" + fs::path(model.dir).filename().string()
                   + "_" + fs::path(model.mesh_a).stem().string()
                   + "_" + fs::path(model.mesh_b).stem().string());
        fs::create_directories(rec.output_root);
#if ISM_HAS_VIEWER
        rec.screenshot_dir = rec.output_root / "screenshots";
        fs::create_directories(rec.screenshot_dir);
#endif

        const fs::path csv_path = rec.output_root / ("metrics_" + suffix + ".csv");
        std::ofstream csv(csv_path);
        csv << "config,stage,corr_mean,corr_median,corr_max,"
               "residual_mean,residual_mean_aw,residual_median,residual_median_aw,"
               "residual_p90,residual_max,residual_frac_gt_0p1,residual_frac_gt_0p1_aw,"
               "residual_area_aw,residual_shape_aw,"
               "energy,energy_per_face,n_faces_T,n_verts_T,n_band_T,"
               "drift_mean_deg,drift_max_deg,"
               "minang_mean_A,minang_worst_A,minang_mean_B,minang_worst_B\n";

        for (const auto& config : configs)
            run_config(model, config, rec.output_root, csv, rec.lifted_As, rec.lifted_Bs, rec.states);

        csv.close();
        ISM_INFO("[" << model.name << "] wrote " << csv_path.string());
    }

    ISM_INFO("======== all " << runs.size() << " model(s) done ========");
    for (const auto& rec : runs)
        ISM_INFO("  " << rec.model.name << "  "
                 << (rec.output_root / ("metrics_" + suffix + ".csv")).string());

#if !ISM_HAS_VIEWER
    // Headless build (e.g. Apple Silicon): macOS caps desktop OpenGL at 4.1 and glow
    // needs 4.2-4.5, so there is no viewer here. The CSVs, the per-config trace and the
    // lifted-T OBJs are all written exactly as above; only the 8 renders are missing.
    // To inspect a per-face field without a window, colour it with linear_colors() and
    // write it with write_colored_ply_flat().
    ISM_INFO("Headless build: skipping the renders (3 per model). CSV, trace and OBJs "
             "are written.");
#else
    // --- Visualization ----------------------------------------------------
    // THREE windows per model, in this order, all of them side-by-side pairs:
    //   Render 0: the sizing fields on mesh A, map-driven (phi) | combined (tels_input)
    //   Render 1: geometry + T wireframe, A | B (does T look sane, do triangles slide?)
    //   Render 2: checkerboard transferred A -> B (sliding and shear are obvious here)
    // Landmarks are drawn into 1 and 2, not as a render of their own -- see
    // LANDMARK_BALLS. Each window blocks until closed, and closing it opens the next.
    //
    // So the window count is 3 x models x configs. With the settled setup --
    // STRETCHED_ONLY and PRESCRIBED_ONLY both on, which leaves exactly one config -- the
    // four MODELS give 12 windows, grouped three at a time per model. Every panel is
    // captioned with the model name, because by window 7 you will not remember.
    //
    // overlay_landmarks() and textured_view() add renderables to whichever gv::view() is
    // currently open, so both have to be called from inside one.
    auto overlay_landmarks = [&] (const MapState& _state, const TriMesh& _lifted_T_i, const int _mesh_idx)
    {
        if (!LANDMARK_BALLS)
            return;

        // Relative to the bounding box, so one factor works whatever the normalization.
        const double radius = LANDMARK_RADIUS_FACTOR * bb_diag(_state.meshes_input[_mesh_idx]);

        // Where the map currently puts each landmark: T's landmark vertex lifted onto the
        // mesh. Drawn first and larger so an input landmark still agreeing with it hides
        // inside it. Empty on a released run.
        if (!_state.landmarks_T.empty())
            view_landmarks(_lifted_T_i, _state.landmarks_T, WidthWorld(radius, true));

        // Where it should be. view_landmarks walks a fresh ColorGenerator in the same
        // order on every call, so landmark j is the same colour on A and on B.
        if (_mesh_idx < (int)_state.landmarks_input.size())
            view_landmarks(_state.meshes_input[_mesh_idx], _state.landmarks_input[_mesh_idx],
                           WidthWorld(0.5f * radius, true));
    };

    // The TEXTURE_SOURCE switch, in one place instead of twice. See the TexSource enum
    // for what the three schemes cost.
    auto textured_view = [&] (const MapState& _state, const int _mesh_idx)
    {
        if (TEXTURE_SOURCE == TexSource::Triplanar)
            view_texture_triplanar_input(_state, 0, _mesh_idx, TEXTURE_FACTOR, texture);
        else if (TEXTURE_SOURCE == TexSource::Sphere)
            view_texture_sphere_embedding_input(_state, 0, _mesh_idx, TEXTURE_FACTOR, texture);
        else
            view_texture_frontal_projection_input(_state, 0, _mesh_idx, TEXTURE_PROJECTION_DIR, TEXTURE_FACTOR, texture);
    };

    // phi for the LEFT panel of render 0. Recomputed here instead of read out of
    // map_tel_phi, so the panel is populated whatever MAP_DRIVEN_TEL was: with the flag
    // off nothing ever fills map_tel_phi. This is a pure function of mesh A and J*, runs
    // after every report() and every write, and touches no part of the map state -- it
    // cannot move the numbers that were just measured. Returns the field the run really
    // used when there is one, so a MAP_DRIVEN_TEL run still shows its own field.
    //
    // The jump threshold is the settings default; the app never overrides it.
    auto phi_field_for = [&] (const MapState& _state)
    {
        ExternalProperty<VH, double> phi;
        if (_state.meshes_input.empty() || _state.prescribed_jacobians.empty())
            return phi; // dirichlet config: no prescription, so no seam to find
        if (_state.map_tel_phi.size_okay(_state.meshes_input[0]))
            return _state.map_tel_phi;

        int n_seam_verts = 0;
        const AdaptiveTriangulationsSettings defaults;
        map_driven_target_edge_lengths(
                    _state.meshes_input[0], _state.prescribed_jacobians[0],
                    defaults.map_tel_jump_thresh, MAP_TEL_FLOOR, MAX_EDGE_LENGTH,
                    MAP_TEL_DIFFUSION_LENGTH, n_seam_verts, nullptr, &phi);
        return phi;
    };

    for (const ModelRun& rec : runs)
    {
        for (int i = 0; i < (int)configs.size(); ++i)
        {
            const std::string name = configs[i].name();
            const std::string tag = rec.model.name + " / " + name;
            const TriMesh* const lifted[2] = { &rec.lifted_As[i], &rec.lifted_Bs[i] };
            const MapState& state = rec.states[i];

            // Render 0: the sizing field on mesh A, map-driven alone | combined.
            //
            // LEFT is phi, the diffused seam indicator on its own: white where the
            // prescription jumps, magenta away from it. RIGHT is tels_input[0], i.e. the
            // field the remesher actually read, which is phi already min()'d with the
            // CURVATURE field -- so on a shape like the L every chamfer is coloured
            // there too. That colouring is curvature, not prescription, and the left
            // panel is the one to judge the map-driven field on.
            //
            // (The left panel still shows some crease colouring of its own, and that
            // part IS the detector: sigma is measured in each face's own tangent frame,
            // so a crease inside a stretched region gives a genuine sigma jump. Measured
            // on the L, 54% of the detected seam edges sit on a crease, all of them at
            // x >= 0.03, i.e. the stretched half.)
            if (!state.meshes_input.empty() && !state.tels_input.empty())
            {
                auto s = screenshot_config(OPEN_VIEWER, rec.screenshot_dir / (name + "_sizing_field.png"), tg::ivec2(1920, 1080), true);
                auto style = default_style();
                auto g = gv::grid();
                const auto phi = phi_field_for(state);
                {
                    auto v = gv::view();
                    view_caption(tag + " : phi (map-driven)");
                    if (phi.size_okay(state.meshes_input[0]))
                        view_vertex_field(state.meshes_input[0], phi, 0.0, 1.0);
                    else
                        view_mesh(state.meshes_input[0], Color(0.8, 0.8, 0.8, 1.0));
                }
                {
                    auto v = gv::view();
                    view_caption(tag + " : tel (combined)");
                    view_vertex_field(state.meshes_input[0], state.tels_input[0],
                                      MAP_TEL_FLOOR, MAX_EDGE_LENGTH);
                }
            }

            // Render 1: the triangulation, lifted onto A | lifted onto B.
            {
                auto s = screenshot_config(OPEN_VIEWER, rec.screenshot_dir / (name + "_wire.png"), tg::ivec2(1920, 1080), true);
                auto style = default_style();
                auto g = gv::grid();
                for (int m = 0; m < 2; ++m)
                {
                    auto v = gv::view();
                    view_caption(tag + (m == 0 ? " : T on A" : " : T on B"));
                    view_mesh(state.meshes_input[m], Color(0.8, 0.8, 0.8, 0.5));
                    view_mesh(*lifted[m], Color(1.0, 1.0, 1.0, 0.8));
                    view_wireframe(*lifted[m], m == 0 ? MAGENTA : TEAL, WidthScreen(0.5));
                    overlay_landmarks(state, *lifted[m], m);
                }
            }

            // Render 2: the checkerboard carried across the map, A | B.
            {
                auto s = screenshot_config(OPEN_VIEWER, rec.screenshot_dir / (name + "_texture.png"), tg::ivec2(1920, 1080), true);
                auto style = default_style();
                auto g = gv::grid();
                for (int m = 0; m < 2; ++m)
                {
                    auto v = gv::view();
                    view_caption(tag + (m == 0 ? " : texture A" : " : texture B"));
                    textured_view(state, m);
                    overlay_landmarks(state, *lifted[m], m);
                }
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
