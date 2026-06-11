 #include <SurfaceMaps/Init.hh>
  #include <SurfaceMaps/Types.hh>
  #include <SurfaceMaps/Utils/IO.hh>
  #include <SurfaceMaps/Utils/MeshNormalization.hh>
  #include <SurfaceMaps/Viewer/MeshView.hh>
  #include <SurfaceMaps/Viewer/HeatmapColors.hh>
  #include <SurfaceMaps/Viewer/Colors.hh>
  #include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>

  #include <Eigen/SVD>

  namespace SurfaceMaps
  {

  void run()
  {
      TriMesh mesh_A = read_mesh(DATA_PATH / "meshes/l_shape/l_shape.obj");
      TriMesh mesh_B = read_mesh(DATA_PATH / "meshes/l_shape/l_shape_extruded.obj");

      // Storage
      ExternalProperty<FH, double> sing_vals_max(mesh_A);
      ExternalProperty<FH, double> sing_vals_min(mesh_A);
      ExternalProperty<FH, Vec3d> vecs_max_A(mesh_A);
      ExternalProperty<FH, Vec3d> vecs_min_A(mesh_A);
      ExternalProperty<FH, Vec3d> vecs_max_B(mesh_A);
      ExternalProperty<FH, Vec3d> vecs_min_B(mesh_A);

      // Scaled vectors for mesh B
      ExternalProperty<FH, Vec3d> vecs_max_B_scaled(mesh_A);
      ExternalProperty<FH, Vec3d> vecs_min_B_scaled(mesh_A);

      TriMesh mesh_A_normalized = mesh_A;
      TriMesh mesh_B_normalized = mesh_B;
      normalize_mesh(mesh_A_normalized);
      normalize_mesh(mesh_B_normalized);

      for (auto fh : mesh_A.faces())
      {
          SVH vh_a, vh_b, vh_c;
          handles(mesh_A, fh, vh_a, vh_b, vh_c);

          Vec3d a_A = mesh_A_normalized.point(vh_a);
          Vec3d b_A = mesh_A_normalized.point(vh_b);
          Vec3d c_A = mesh_A_normalized.point(vh_c);

          Vec3d a_B = mesh_B_normalized.point(vh_a);
          Vec3d b_B = mesh_B_normalized.point(vh_b);
          Vec3d c_B = mesh_B_normalized.point(vh_c);

          // Local bases
          Vec3d normal_A = ((b_A - a_A).cross(c_A - a_A)).normalized();
          Vec3d basis0_A = (b_A - a_A).normalized();
          Vec3d basis1_A = normal_A.cross(basis0_A);

          Vec3d normal_B = ((b_B - a_B).cross(c_B - a_B)).normalized();
          Vec3d basis0_B = (b_B - a_B).normalized();
          Vec3d basis1_B = normal_B.cross(basis0_B);

          // Local coordinates
          Eigen::Vector2<double> a_local_A, b_local_A, c_local_A;
          Eigen::Vector2<double> a_local_B, b_local_B, c_local_B;
          to_local_coordinates(a_A, b_A, c_A, a_local_A, b_local_A, c_local_A);
          to_local_coordinates(a_B, b_B, c_B, a_local_B, b_local_B, c_local_B);

          Eigen::Matrix2<double> M_A;
          M_A << b_local_A - a_local_A, c_local_A - a_local_A;
          Eigen::Matrix2<double> M_B;
          M_B << b_local_B - a_local_B, c_local_B - a_local_B;

          Eigen::Matrix2<double> J = M_B * M_A.inverse();
          Eigen::JacobiSVD<Eigen::Matrix2<double>> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);

          double s_max = svd.singularValues()[0];
          double s_min = svd.singularValues()[1];
          sing_vals_max[fh] = s_max;
          sing_vals_min[fh] = s_min;

          Eigen::Vector2<double> v1_2d = svd.matrixV().col(0);
          Eigen::Vector2<double> v2_2d = svd.matrixV().col(1);
          Eigen::Vector2<double> u1_2d = svd.matrixU().col(0);
          Eigen::Vector2<double> u2_2d = svd.matrixU().col(1);

          // Unscaled vectors (unit length)
          vecs_max_A[fh] = v1_2d[0] * basis0_A + v1_2d[1] * basis1_A;
          vecs_min_A[fh] = v2_2d[0] * basis0_A + v2_2d[1] * basis1_A;
          vecs_max_B[fh] = u1_2d[0] * basis0_B + u1_2d[1] * basis1_B;
          vecs_min_B[fh] = u2_2d[0] * basis0_B + u2_2d[1] * basis1_B;

          // Scaled vectors for mesh B (by singular values)
          vecs_max_B_scaled[fh] = s_max * vecs_max_B[fh];
          vecs_min_B_scaled[fh] = s_min * vecs_min_B[fh];
      }

      // Visualization: side-by-side
      {
          auto style = default_style();
          auto g = gv::grid();  // Creates side-by-side layout

          // Left: Mesh A with frame field (unscaled)
          {
              auto v = gv::view();
              view_mesh(mesh_A);
              view_vector_field(mesh_A, vecs_max_A, 0.05, RED);
              view_vector_field(mesh_A, vecs_min_A, 0.05, BLUE);
          }

          // Right: Mesh B with frame field (scaled by singular values)
          {
              auto v = gv::view();
              view_mesh(mesh_B);
              view_vector_field(mesh_B, vecs_max_B_scaled, 0.05, RED);
              view_vector_field(mesh_B, vecs_min_B_scaled, 0.05, BLUE);
          }
      }
  }

  }

  int main()
  {
      glow::glfw::GlfwContext ctx;
      using namespace SurfaceMaps;
      init_lib_surface_maps();
      run();
      return 0;
  }