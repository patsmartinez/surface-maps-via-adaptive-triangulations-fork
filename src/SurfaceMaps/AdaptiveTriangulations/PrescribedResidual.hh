/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Residual of the realized map against a prescribed Jacobian field.
 *
 * This lives in the library rather than in an app because it MUST mirror
 * eval_trianglepair_energy_triangle_T: it has to lift through the same two sphere
 * embeddings the energy lifts through. While it was a private copy inside
 * baseline_comparison.cc it silently drifted -- it lifted the mesh-A sphere position onto
 * BOTH meshes, so it evaluated emb_B^-1 o emb_A (the initial map, resampled on the
 * current T) and was blind to every map DOF the optimizer touches. That is why the
 * residual "never moved" in any experiment, and why the A-on-A residual sat at 2e-16
 * while corr_mean drifted to 0.0245. One definition, used by every app, so it cannot
 * happen again.
 */
#pragma once

#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>

#include <utility>
#include <vector>

namespace SurfaceMaps
{

/// Distribution of the per-face residual.
///
/// Two things the plain mean cannot do:
///
/// (1) COMPARE ACROSS STAGES. The unweighted average over T faces changes meaning when
///     refinement takes T from ~50 faces to ~630, because refinement concentrates small
///     faces wherever it chooses to refine. The *_aw fields weight each face by the area
///     of the mesh-A material it covers, which is stage-independent; compare those.
///     The unweighted fields are kept unchanged so older CSVs remain comparable, and the
///     gap between `mean` and `mean_aw` is itself a readout of how much the unweighted
///     statistic is being distorted.
///
/// (2) SAY WHAT KIND OF WRONG. The residual J J*^-1 has singular values s0, s1, and
///     log-splits into two independent parts:
///         area  = |log(s0*s1)|  -- material is in the wrong PLACE (mis-allocated area)
///         shape = |log(s0/s1)|  -- material is the wrong SHAPE   (sheared / anisotropy off)
///     Both vanish exactly at s0 = s1 = 1. Area-dominant points at coarse-scale
///     representativeness (the centroid lookup smearing the prescription); shape-dominant
///     points at local shear, i.e. sliding and initialization.
struct ResidualStats
{
    double mean = 0.0;            ///< unweighted; unchanged, comparable with earlier runs
    double mean_aw = 0.0;         ///< area-weighted -- the one to compare ACROSS stages
    double median = 0.0;
    double median_aw = 0.0;
    double p90 = 0.0;
    double max = 0.0;             ///< per-face max over the two singular values, then max over faces
    double frac_gt_0p1 = 0.0;     ///< fraction of faces off by more than 10%
    double frac_gt_0p1_aw = 0.0;  ///< fraction of AREA off by more than 10%
    double area_part_aw = 0.0;    ///< area-weighted mean |log(s0*s1)|
    double shape_part_aw = 0.0;   ///< area-weighted mean |log(s0/s1)|
    int n_faces = 0;              ///< faces actually measured
    int n_faces_skipped = 0;      ///< faces dropped: flipped on either side, or sigma ~ 0
    double area_measured = 0.0;   ///< total mesh-A area behind the measured faces

    /// Fraction of T's faces that could be measured at all. A search that MINIMIZES the
    /// residual can cheat by flipping faces, because a flipped face is skipped and so
    /// leaves the mean -- always check or penalize this.
    double measured_fraction() const
    {
        const int total = n_faces + n_faces_skipped;
        return (total > 0) ? (double)n_faces / (double)total : 0.0;
    }
};

/// Percentile of (value, weight) pairs, sorted ascending by value.
double weighted_percentile(
        const std::vector<std::pair<double, double>>& _sorted,
        const double _total_weight,
        const double _q);

/// Rotation-invariant residual against the prescribed field: singular values of
/// J J*^-1, which are all 1 exactly when the realized pullback metric matches the
/// prescription. Meaningful for Dirichlet runs too -- they do not optimize it, but they
/// are still evaluated against it.
///
/// @param _pair_idx Index into pairs_map_distortion / prescribed_jacobians.
void compute_metric_residual(
        const MapState& _map_state,
        ResidualStats& _stats,
        const int _pair_idx = 0);

}
