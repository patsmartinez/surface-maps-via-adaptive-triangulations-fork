#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/AdaptiveTriangulations/EvaluationMetrics.hh>
#include <SurfaceMaps/Viewer/HeatmapColors.hh>
#include <fstream>
#include <SurfaceMaps/Viewer/Colors.hh>

namespace SurfaceMaps
  {

void write_colored_ply(
      const TriMesh& mesh,
      const ExternalProperty<FH, Color>& face_colors,
      const fs::path& path)
  {
      // Convert face colors to vertex colors (average of adjacent faces)
      std::vector<Color> vertex_colors(mesh.n_vertices(), Color(0,0,0,0));
      std::vector<int> vertex_counts(mesh.n_vertices(), 0);

      for (auto fh : mesh.faces()) {
          Color c = face_colors[fh];
          for (auto vh : fh.vertices()) {
              vertex_colors[vh.idx()] += c;
              vertex_counts[vh.idx()]++;
          }
      }

      for (size_t i = 0; i < vertex_colors.size(); i++) {
          if (vertex_counts[i] > 0) {
              vertex_colors[i] /= (double)vertex_counts[i];
          }
      }

      std::ofstream file(path);

      file << "ply\n";
      file << "format ascii 1.0\n";
      file << "element vertex " << mesh.n_vertices() << "\n";
      file << "property float x\n";
      file << "property float y\n";
      file << "property float z\n";
      file << "property uchar red\n";
      file << "property uchar green\n";
      file << "property uchar blue\n";
      file << "element face " << mesh.n_faces() << "\n";
      file << "property list uchar int vertex_indices\n";
      file << "end_header\n";

      int i = 0;
      for (auto vh : mesh.vertices()) {
          auto p = mesh.point(vh);
          Color c = vertex_colors[i++];
          file << p[0] << " " << p[1] << " " << p[2] << " "
               << (int)(c[0]*255) << " " << (int)(c[1]*255) << " " << (int)(c[2]*255) << "\n";
      }

      for (auto fh : mesh.faces()) {
          file << "3";
          for (auto vh : fh.vertices()) {
              file << " " << vh.idx();
          }
          file << "\n";
      }

      ISM_INFO("Wrote " << path);
  }

void visualize_distortion(const TriMesh& mesh_1, const TriMesh& mesh_2)
  {
    ISM_INFO("Mesh 1: " << mesh_1.n_vertices() << " verts, " << mesh_1.n_faces() << " faces");
    ISM_INFO("Mesh 2: " << mesh_2.n_vertices() << " verts, " << mesh_2.n_faces() << " faces");

    ExternalProperty<FH, double> areas_1, areas_2, sing_vals_min, sing_vals_max;
    mapping_distortion_normalized(mesh_1, mesh_2, areas_1, areas_2, sing_vals_min, sing_vals_max);

    ExternalProperty<FH, double> distortion_metric(mesh_1);
    for (auto fh : mesh_1.faces()) {
        distortion_metric[fh] = sing_vals_max[fh];
        // Or: distortion_metric[fh] = sing_vals_max[fh] / sing_vals_min[fh];
    }
    // After computing distortion_metric:
    double min_dist = 1e10, max_dist = 0;
    for (auto fh : mesh_1.faces()) {
        min_dist = std::min(min_dist, distortion_metric[fh]);
        max_dist = std::max(max_dist, distortion_metric[fh]);
    }
    ISM_INFO("Distortion range: " << min_dist << " to " << max_dist);

    // Then set your color range based on actual data:
    double min_val = min_dist;
    double max_val = max_dist;
    
    auto colors = linear_colors(distortion_metric, min_val, max_val, WHITE, MAGENTA);

    write_colored_ply(mesh_2, colors, OUTPUT_PATH / "distortion.ply");
}

}

int main()
{
    // NO GlfwContext needed!
    using namespace SurfaceMaps;
    TriMesh mesh_1 = read_mesh(DATA_PATH / "meshes/l_shape/l_shape.obj");
    TriMesh mesh_2 = read_mesh(DATA_PATH / "meshes/l_shape/l_shape_extruded.obj");

    visualize_distortion(mesh_1, mesh_2);

    return 0;
  }