
#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/Viewer/HeatmapColors.hh>
#include <SurfaceMaps/Viewer/MeshView.hh>

namespace SurfaceMaps
{

void visualize_distortion(const TriMesh& mesh_1, const TriMesh& mesh_2){

    ExternalProperty<FH, double> areas_1, areas_2, sing_vals_min, sing_vals_max;
    mapping_distortion_normalized(mesh_1, mesh_2, areas_1, areas_2, sing_vals_min, sing_vals_max);

    ExternalProperty<FH, double> distortion_metric(mesh_1);

    for (auto fh : mesh_1.faces()){
        distortion_metric[fh] = sing_vals_max[fh];

        // distortion_metric[fh] = sing_vals_max[fh] / sing_vals_min[fh];

    }

    double min_val = 1.0;
    double max_val = 4.0;

    auto colors = linear_colors(distortion_metric, min_val, max_val, WHITE, MAGENTA);

    view_face_colors(mesh_1, colors);

}
}


int main()
{

    glow::glfw::GlfwContext ctx;
    using namespace SurfaceMaps;
    init_lib_surface_maps();

    //Load meshes
    TriMesh L_mesh, L_stretched_mesh;
    read_mesh(DATA_PATH / "meshes/l_shape/l_shape.obj");
    read_mesh(DATA_PATH / "meshes/l_shape/l_shape_extruded.obj");

    visualize_distortion(L_mesh, L_stretched_mesh);
    return 0;
}