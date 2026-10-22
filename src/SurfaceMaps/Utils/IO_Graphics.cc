/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 *
 * Authors: Patrick Schmidt, Janis Born
 *
 * The parts of IO.cc that need polymesh / glow / typed-geometry, split out so that
 * IO.cc -- which every app reaches through IO.hh -- compiles without a graphics
 * stack. The headless build target excludes this translation unit; the declarations
 * stay in IO.hh either way, which is harmless as long as nothing headless calls them.
 */
#include "IO.hh"

#include <polymesh/Mesh.hh>
#include <glow/objects/Texture2D.hh>
#include <SurfaceMaps/Utils/Helpers.hh>

namespace SurfaceMaps
{

namespace
{

template <typename OM_Handle, typename PM_Handle>
std::vector<OM_Handle> openmesh_handles(
        const std::vector<PM_Handle>& _pm)
{
    std::vector<OM_Handle> om;
    for (const auto& h : _pm)
        om.push_back(OM_Handle(h.idx.value));

    return om;
}

template <typename PM_Handle, typename OM_Handle>
std::vector<PM_Handle> polymesh_handles(
        const std::vector<OM_Handle>& _om,
        const pm::Mesh& _mesh)
{
    std::vector<PM_Handle> pm;
    for (const auto& h : _om)
        pm.push_back(_mesh.handle_of(typename PM_Handle::index_t(h.idx())));

    return pm;
}

}

std::vector<pm::edge_handle> read_paths(
        const std::string& _path,
        const pm::Mesh& _mesh,
        const bool _throw_if_not_exist)
{
    return polymesh_handles<pm::edge_handle>(read_paths(_path, _throw_if_not_exist), _mesh);
}

void write_paths(
        const std::vector<pm::edge_handle>& _edges,
        const std::string& _path,
        const bool _overwrite)
{
    // Goes through the OpenMesh overload rather than the file-local
    // write_element_handles template, which stayed behind in IO.cc.
    write_paths(openmesh_handles<EH>(_edges), _path, _overwrite);
}

glow::SharedTexture2D read_texture(
        const fs::path &_file_path)
{
    return glow::Texture2D::createFromFile(_file_path, glow::ColorSpace::sRGB);
}

pm::vertex_attribute<Vec3d> to_polymesh(
        const TriMesh& mesh,
        pm::Mesh& m)
{
    for (auto vh : mesh.vertices())
        m.vertices().add();
    for (auto fh : mesh.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(mesh, fh, vh_a, vh_b, vh_c);
        m.faces().add(m.handle_of(pm::vertex_index(vh_b.idx())),
                      m.handle_of(pm::vertex_index(vh_c.idx())),
                      m.handle_of(pm::vertex_index(vh_a.idx())));
    }

    auto pos = m.vertices().make_attribute<Vec3d>();
    for (auto vh : mesh.vertices())
        pos[m.handle_of(pm::vertex_index(vh.idx()))] = mesh.point(vh);

    return pos;
}

pm::vertex_attribute<tg::pos3> to_polymesh_tg(
        const TriMesh& mesh,
        pm::Mesh& m)
{
    for (auto vh : mesh.vertices())
        m.vertices().add();
    for (auto fh : mesh.faces())
    {
        VH vh_a, vh_b, vh_c;
        handles(mesh, fh, vh_a, vh_b, vh_c);
        m.faces().add(m.handle_of(pm::vertex_index(vh_b.idx())),
                      m.handle_of(pm::vertex_index(vh_c.idx())),
                      m.handle_of(pm::vertex_index(vh_a.idx())));
    }

    auto pos = m.vertices().make_attribute<tg::pos3>();
    for (auto vh : mesh.vertices())
        pos[m.handle_of(pm::vertex_index(vh.idx()))] = tg::pos3(mesh.point(vh));

    return pos;
}

}
