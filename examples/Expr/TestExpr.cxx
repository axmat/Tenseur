#include "Ten/Storage/DenseStorage.hxx"
#include <ios>
#include <iostream>

#include <Ten/Expr.hxx>
#include <Ten/Functional.hxx>
#include <Ten/Shape.hxx>
#include <Ten/Tensor.hxx>
#include <memory>
#include <type_traits>

template <class T> void printTensor(const T &val) {
   std::cout << "[";
   for (size_t i = 0; i < val.size(); i++)
      std::cout << val[i] << " ";
   std::cout << "]\n";
}

int main() {
   using namespace ten;
   using namespace std;

   {
      cout << "UnaryExpr" << endl;
      DynamicTensor<float, 1> x({3});
      for (size_t i = 0; i < 3; i++)
         x[i] = -float(i);

      auto e = abs(x).eval();

      cout << "abs(x) = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << e[i] << " ";
      cout << "]" << endl;

      cout << "And x = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << x[i] << " ";
      cout << "]" << endl;
   }

   {
      cout << "UnaryExpr min" << endl;
      DynamicTensor<float, 1> x({3});
      for (size_t i = 0; i < 3; i++)
         x[i] = -float(i + 1.);
      auto e = min(x);
      auto v = e.eval();

      cout << "min(x) = [ ";
      float m = x[0];
      for (size_t i = 1; i < 3; i++) {
         if (x[i] < m)
            m = x[i];
      }
      cout << m << " ";
      cout << "]" << endl;

      cout << "And expr value = " << v.value() << endl;
   }

   {
      cout << "UnaryExpr sqrt" << endl;
      DynamicTensor<float, 1> x({3});
      for (size_t i = 0; i < 3; i++)
         x[i] = float(i);
      using Tensor_t = DynamicTensor<float, 1>;
      auto e = sqrt(x);
      auto out = e.eval();

      cout << "sqrt(x) = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << out[i] << " ";
      cout << "]" << endl;

      cout << "And x = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << x[i] << " ";
      cout << "]" << endl;
   }

   {
      cout << "Binary expr a + b" << std::endl;
      DynamicTensor<float, 1> a({3});
      DynamicTensor<float, 1> b({3});
      for (size_t i = 0; i < 3; i++) {
         a[i] = i;
         b[i] = i;
      }

      auto c = a + b;
      auto res = c.eval();
      printTensor(a);
      printTensor(b);
      printTensor(res);
   }

   {
      cout << "Binary expr a - b" << std::endl;
      DynamicTensor<float, 1> a({3});
      DynamicTensor<float, 1> b({3});
      for (size_t i = 0; i < 3; i++) {
         a[i] = i;
         b[i] = i + 1;
      }
      auto c = (a - b).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), DynamicTensor<float, 1>>);
   }

   {
      cout << "Binary expr a * b" << std::endl;
      DynamicTensor<float, 1> a({6});
      DynamicTensor<float, 1> b({6});
      for (size_t i = 0; i < 6; i++) {
         a[i] = i;
         b[i] = i;
      }
      auto c = (a * b).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), DynamicTensor<float, 1>>);
   }

   {
      cout << "Binary expr A * B" << std::endl;
      DynamicTensor<float, 2> a({2, 3});
      DynamicTensor<float, 2> b({3, 4});
      for (size_t i = 0; i < 2 * 3; i++) {
         a[i] = i;
      }
      for (size_t i = 0; i < 3 * 4; i++) {
         b[i] = i;
      }
      auto c = (a * b).eval();
      cout << "A = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            std::cout << a(i, j) << " ";
         }
         cout << endl;
      }
      cout << "B = \n";
      for (size_t i = 0; i < 3; i++) {
         for (size_t j = 0; j < 4; j++) {
            std::cout << b(i, j) << " ";
         }
         cout << endl;
      }
      cout << "C = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 4; j++) {
            std::cout << c(i, j) << " ";
         }
         cout << endl;
      }
      static_assert(std::is_same_v<decltype(c), DynamicTensor<float, 2>>);
   }

   {
      cout << "Binary expr matrix * vector" << std::endl;
      DynamicTensor<float, 2> a({2, 3});
      DynamicTensor<float, 1> b({3});
      for (size_t i = 0; i < 2 * 3; i++) {
         a[i] = i;
      }
      for (size_t i = 0; i < 3; i++) {
         b[i] = i;
      }
      auto c = (a * b).eval();
      cout << "A = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            std::cout << a(i, j) << " ";
         }
         cout << endl;
      }
      cout << "B = \n";
      for (size_t j = 0; j < 3; j++) {
         std::cout << b[j] << " ";
      }
      cout << endl;
      cout << "C = \n";
      for (size_t j = 0; j < 2; j++) {
         std::cout << c[j] << " ";
      }
      cout << endl;
      static_assert(std::is_same_v<decltype(c), DynamicTensor<float, 1>>);
   }

   {
      cout << "Binary expr alpha * a" << std::endl;
      DynamicTensor<float, 1> a({5});
      for (size_t i = 0; i < 5; i++) {
         a[i] = i;
      }
      auto c = (2. * a).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), DynamicTensor<float, 1>>);
   }

   {
      cout << "Chain binary expressions" << std::endl;
      DynamicTensor<float, 1> a({5}), b({5});
      for (size_t i = 0; i < 5; i++) {
         a[i] = i;
         b[i] = i;
      }
      auto c = a + b;
      // 0 2 4  6  8
      // 0 2 8 18 32
      auto d = (c * a).eval();
      printTensor(d);
   }

   {
      cout << "Chain unary expressions" << std::endl;
      DynamicTensor<float, 1> a({5});
      for (size_t i = 0; i < 5; i++) {
         a[i] = i;
      }
      auto b = sqrt(a);
      auto c = sqrt(b).eval();
      printTensor(c);
   }

   {
      cout << "Chain unary and binary expressions" << std::endl;
      DynamicTensor<float, 1> a({5});
      for (size_t i = 0; i < 5; i++) {
         a[i] = i;
      }
      auto b = 2. * a;
      auto c = sqrt(b);
      auto d = (c + a).eval();
      printTensor(d);
   }

   {
      cout << "Tensor from unary expr" << std::endl;
      auto a = iota<5>();
      Tensor<float, Shape<5>> b = sqrt(a);
      printTensor(b);
   }

   {
      cout << "Tensor from binary expr" << std::endl;
      auto a = iota<5>();
      auto b = iota<5>();
      Tensor<float, Shape<5>> d = a + b;
      printTensor(d);
   }

   return 0;
}
