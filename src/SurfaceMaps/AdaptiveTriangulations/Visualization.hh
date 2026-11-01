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
#include <SurfaceMaps/Utils/Filesystem.hh>
#include <typed-geometry/types/vec.hh>
#include <SurfaceMaps/AdaptiveTriangulations/MapState.hh>
#include <glow-extras/viewer/configure.hh>

namespace SurfaceMaps
{

/// view texture via frontal projection on lifted T_A and T_B
void view_texture_frontal_projection_T(
        const TriMesh& _mesh_T_A,
        const TriMesh& _mesh_T_B,
        const std::string& _name="",
        const int& _projection_dir = 2,
        const double& _texture_factor=2.0,
        const fs::path& _texture_path = DATA_PATH / "textures/checkerboard.png",
        const fs::path& screenshot_path="",
        const tg::ivec2& _size = tg::ivec2(1920, 1080),
        const bool _transparent = false,
        const glow::viewer::camera_transform& _cam_pos = glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

/// view texture via frontal projection on lifted T_A and T_B (MapState)
void view_texture_frontal_projection_T(
        const MapState& _map_state,
        const int _ref_mesh_num = 0,
        const std::vector<int>& _indices = {},
        const int& _projection_dir = 2,
        const double& _texture_factor = 2.0,
        const fs::path& _texture_path = DATA_PATH / "textures/checkerboard.png",
        const fs::path& screenshot_path = "",
        const tg::ivec2& _size = tg::ivec2(1920, 1080),
        const bool _transparent = false,
        const glow::viewer::camera_transform& _cam_pos = glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

/// Checkerboard laid out on T's SPHERE EMBEDDING instead of projected along a world
/// axis, so it does not degenerate on faces parallel to that axis. See
/// sphere_embedding_texcoords (Helpers.hh) for why the frontal projection fails on an
/// extruded shape and what the seam/pole trade-off is.
///
/// One texcoord array, built from mesh `_ref_mesh_num`'s sphere positions, is applied to
/// every lifted mesh -- so matching checkers still mean matching material, which is the
/// whole point of the comparison.
void view_texture_sphere_embedding_T(
        const MapState& _map_state,
        const int _ref_mesh_num = 0,
        const double& _texture_factor = 2.0,
        const fs::path& _texture_path = DATA_PATH / "textures/checkerboard.png",
        const fs::path& _screenshot_path = "",
        const tg::ivec2& _size = tg::ivec2(1920, 1080),
        const bool _transparent = false,
        const glow::viewer::camera_transform& _cam_pos = glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

/// Sphere-embedding counterpart of view_texture_frontal_projection_input: lay the
/// checkerboard out on the SOURCE mesh's sphere embedding, transfer it through the map,
/// and draw it on the target INPUT mesh. This is the view that shows sliding and shear
/// on the original geometry; see sphere_embedding_texcoords for why the planar
/// projection degenerates on an extruded shape.
void view_texture_sphere_embedding_input(
        const MapState& _map_state,
        const int _source_mesh_idx,
        const int _target_mesh_idx,
        const double& _texture_factor,
        const glow::SharedTexture2D& _texture);

/// Show texture mapped to target mesh
void view_texture_frontal_projection_input(
        const MapState& _map_state,
        const int _source_mesh_idx,
        const int _target_mesh_idx,
        const int &_projection_dir,
        const double& _texture_factor,
        const glow::SharedTexture2D& _texture);

/// view texture via frontal projection on _ref_mesh_num mesh on all input meshees (MapState)
void view_texture_frontal_projection_input(
        const MapState& _map_state,
        const int _ref_mesh_num = 0,
        const std::vector<int>& _indices = {},
        const int& _projection_dir = 2,
        const double& _texture_factor = 2.0,
        const fs::path& _texture_path = DATA_PATH / "textures/checkerboard.png",
        const fs::path& screenshot_path = "",
        const tg::ivec2& _size = tg::ivec2(1920, 1080),
        const bool _transparent = false,
        const glow::viewer::camera_transform& _cam_pos = glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

/// Show texture (defined by halfedgetexcoords of source mesh) mapped to target mesh
void view_texture_halfedgetexcoords_input(
        const MapState& _map_state,
        const int _source_mesh_idx,
        const int _target_mesh_idx,
        const glow::SharedTexture2D& _texture);

/// view lifted wireframes with landmark points (MapState)
void view_lifted_with_landmarks_in_meshes(
        const MapState& _map_state,
        const fs::path& _screenshot_path = "",
        const tg::ivec2& _size= tg::ivec2(1920, 1080),
        const bool _transparent= false,
        const glow::viewer::camera_transform& _cam_pos= glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

void view_landmarks_on_spheres(
        const MapState& _map_state,
        const fs::path& _screenshot_path = "",
        const tg::ivec2& _size= tg::ivec2(1920, 1080),
        const bool _transparent= false,
        const glow::viewer::camera_transform& _cam_pos= glow::viewer::camera_transform(tg::pos3(-0.081837f, 0.306927f, 2.645693f), tg::pos3(0.f, 0.f, 0.f)));

}
