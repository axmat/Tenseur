#include <fstream>
#include <iostream>
#include <nanobench.h>

#include <Eigen/Core>
using namespace Eigen;

int main(int argc, char **argv) {
   if (argc > 2) {
      std::cerr << "./EigenBench [file_name]" << std::endl;
      return 1;
   }
   std::string fileName = (argc == 2) ? std::string(argv[1]) : "eigenBench";

   ankerl::nanobench::Bench bench;
   bench.title("Eigen");

   std::vector<size_t> sizes = {512, 1024, 2048};

   for (auto N : sizes) {
      MatrixXf a(N, N);
      MatrixXf b(N, N);
      MatrixXf c = MatrixXf::Zero(N, N);
      size_t k = 0;
      for (size_t i = 0; i < N; i++) {
         for (size_t j = 0; j < N; j++) {
            a(j, i) = k;
            b(j, i) = k;
            k++;
         }
      }

      bench.run("Gemm", [&] { c = a * b; });
      bench.run("Gemm2", [&] { MatrixXf d = a * b; });
   }

   std::ofstream file(fileName + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   return 0;
}