#include <SurfaceMaps/Types.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>

#include <Eigen/SVD>

namespace SurfaceMaps
{

void mapping_distortion_with_vectors(
          const TriMesh& _mesh_TA,
          const TriMesh& _mesh_TB,
          ExternalProperty<FH, double>& _sing_vals_min,
          ExternalProperty<FH, double>& _sing_vals_max,
          ExternalProperty<FH, Vec3d>& _sing_vecs_max_A,
          ExternalProperty<FH, Vec3d>& _sing_vecs_min_A,
          ExternalProperty<FH, Vec3d>& _sing_vecs_max_B,
          ExternalProperty<FH, Vec3d>& _sing_vecs_min_B)
{
      ISM_ASSERT_EQ(_mesh_TA.n_faces(), _mesh_TB.n_faces());

      _sing_vals_min.init(_mesh_TA);
      _sing_vals_max.init(_mesh_TA);
      _sing_vecs_max_A.init(_mesh_TA);
      _sing_vecs_min_A.init(_mesh_TA);
      _sing_vecs_max_B.init(_mesh_TA);
      _sing_vecs_min_B.init(_mesh_TA);

      TriMesh mesh_A_normalized = _mesh_TA;
      TriMesh mesh_B_normalized = _mesh_TB;

      normalize_mesh(mesh_A_normalized);
      normalize_mesh(mesh_B_normalized);

      for (auto fh : _mesh_TA.faces())
      {
          SVH vh_a, vh_b, vh_c;
          handles(_mesh_TA, fh, vh_a, vh_b, vh_c);

          Vec3d a_A = mesh_A_normalized.point(vh_a);
          Vec3d b_A = mesh_A_normalized.point(vh_b);
          Vec3d c_A = mesh_A_normalized.point(vh_c);

          Vec3d a_B = mesh_B_normalized.point(vh_a);
          Vec3d b_B = mesh_B_normalized.point(vh_b);
          Vec3d c_B = mesh_B_normalized.point(vh_c);

          // Local basis for mesh A
          Vec3d normal_A = ((b_A - a_A).cross(c_A - a_A)).normalized();
          Vec3d basis0_A = (b_A - a_A).normalized();
          Vec3d basis1_A = normal_A.cross(basis0_A);

          // Local basis for mesh B
          Vec3d normal_B = ((b_B - a_B).cross(c_B - a_B)).normalized();
          Vec3d basis0_B = (b_B - a_B).normalized();
          Vec3d basis1_B = normal_B.cross(basis0_B);

          // Local coordinate system per face for lifted vertices
          Eigen::Vector2<double> a_lifted_local_A, b_lifted_local_A, c_lifted_local_A;
          Eigen::Vector2<double> a_lifted_local_B, b_lifted_local_B, c_lifted_local_B;
          to_local_coordinates(a_A, b_A, c_A, a_lifted_local_A, b_lifted_local_A, c_lifted_local_A);
          to_local_coordinates(a_B, b_B, c_B, a_lifted_local_B, b_lifted_local_B, c_lifted_local_B);

          Eigen::Matrix2<double> M_A;
          M_A << b_lifted_local_A - a_lifted_local_A, c_lifted_local_A - a_lifted_local_A;
          Eigen::Matrix2<double> M_B;
          M_B << b_lifted_local_B - a_lifted_local_B, c_lifted_local_B - a_lifted_local_B;

          // Jacobian
          Eigen::Matrix2<double> J = M_B * M_A.inverse();
          Eigen::JacobiSVD<Eigen::Matrix2<double>> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);

          _sing_vals_min[fh] = svd.singularValues()[1];
          _sing_vals_max[fh] = svd.singularValues()[0];
          ISM_ASSERT_LEQ(_sing_vals_min[fh], _sing_vals_max[fh]);

          // Extract singular vectors (2D)
          Eigen::Vector2<double> v1_2d = svd.matrixV().col(0);
          Eigen::Vector2<double> v2_2d = svd.matrixV().col(1);
          Eigen::Vector2<double> u1_2d = svd.matrixU().col(0);
          Eigen::Vector2<double> u2_2d = svd.matrixU().col(1);

          // Convert to 3D using local bases
          _sing_vecs_max_A[fh] = v1_2d[0] * basis0_A + v1_2d[1] * basis1_A;
          _sing_vecs_min_A[fh] = v2_2d[0] * basis0_A + v2_2d[1] * basis1_A;
          _sing_vecs_max_B[fh] = u1_2d[0] * basis0_B + u1_2d[1] * basis1_B;
          _sing_vecs_min_B[fh] = u2_2d[0] * basis0_B + u2_2d[1] * basis1_B;
      }
  }

  void write_mapping_distortion_with_vectors(
          const TriMesh& _mesh_TA,
          const TriMesh& _mesh_TB,
          const fs::path& _csv_path)
  {
      ISM_ASSERT_EQ(_mesh_TA.n_faces(), _mesh_TB.n_faces());

      ExternalProperty<FH, double> sing_vals_min;
      ExternalProperty<FH, double> sing_vals_max;
      ExternalProperty<FH, Vec3d> sing_vecs_max_A;
      ExternalProperty<FH, Vec3d> sing_vecs_min_A;
      ExternalProperty<FH, Vec3d> sing_vecs_max_B;
      ExternalProperty<FH, Vec3d> sing_vecs_min_B;

      mapping_distortion_with_vectors(_mesh_TA, _mesh_TB,
                                       sing_vals_min, sing_vals_max,
                                       sing_vecs_max_A, sing_vecs_min_A,
                                       sing_vecs_max_B, sing_vecs_min_B);

      fs::remove_all(_csv_path);
      append_to_csv(_csv_path,
                    "sing_val_max", "sing_val_min",
                    "vec_max_A_x", "vec_max_A_y", "vec_max_A_z",
                    "vec_min_A_x", "vec_min_A_y", "vec_min_A_z",
                    "vec_max_B_x", "vec_max_B_y", "vec_max_B_z",
                    "vec_min_B_x", "vec_min_B_y", "vec_min_B_z");

      for (auto f : _mesh_TA.faces())
      {
          append_to_csv_silent(_csv_path,
                               sing_vals_max[f], sing_vals_min[f],
                               sing_vecs_max_A[f][0], sing_vecs_max_A[f][1], sing_vecs_max_A[f][2],
                               sing_vecs_min_A[f][0], sing_vecs_min_A[f][1], sing_vecs_min_A[f][2],
                               sing_vecs_max_B[f][0], sing_vecs_max_B[f][1], sing_vecs_max_B[f][2],
                               sing_vecs_min_B[f][0], sing_vecs_min_B[f][1], sing_vecs_min_B[f][2]);
      }

      ISM_INFO("Wrote SVD data to " << _csv_path);
  }

  void run()
  {
      const fs::path mesh_path_A = DATA_PATH / "meshes/l_shape/l_shape.obj";
      const fs::path mesh_path_B = DATA_PATH / "meshes/l_shape/l_shape_extruded.obj";

      TriMesh mesh_A = read_mesh(mesh_path_A);
      TriMesh mesh_B = read_mesh(mesh_path_B);

      ISM_INFO("Mesh A: " << mesh_A.n_vertices() << " verts, " << mesh_A.n_faces() << " faces");
      ISM_INFO("Mesh B: " << mesh_B.n_vertices() << " verts, " << mesh_B.n_faces() << " faces");

      fs::path output_dir = OUTPUT_PATH / "svd_visualization";
      fs::create_directories(output_dir);

      write_mapping_distortion_with_vectors(mesh_A, mesh_B, output_dir / "svd_data.csv");
  }

  }

  int main()
  {
      using namespace SurfaceMaps;
      run();
      return 0;
  }