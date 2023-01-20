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

      using Tensor_t = DynamicTensor<float, 1>;

      static_assert(std::is_same_v<
                    Tensor_t::node_type,
                    TensorNode<float, DynamicShape<1>, StorageOrder::ColMajor,
                               DenseStorage<float, std::allocator<float>>,
                               std::allocator<float>>>);

      auto e = abs(x);
      using E = decltype(e);
      static_assert(std::is_same_v<E, UnaryExpr<Tensor_t, functional::Abs>>);
      static_assert(std::is_same_v<E::parent_type, Tensor_t::node_type>);

      static_assert(
          std::is_same_v<E::node_type,
                         UnaryNode<Tensor_t::node_type, Tensor_t::node_type,
                                   functional::Abs>>);

      static_assert(std::is_same_v<E::input_type, Tensor_t::node_type>);

      static_assert(std::is_same_v<E::output_type, Tensor_t::node_type>);

      cout << x.node().get() << endl;
      cout << e.parent().get() << endl;

      e.eval();
      auto out_node = e.valueNode();
      auto out_tensor = e.value();

      cout << "abs(x) = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << out_tensor[i] << " ";
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
      using Tensor_t = DynamicTensor<float, 1>;

      using t = InputNode<Tensor_t, ten::functional::Min>;
      using r = t::type;
      static_assert(std::is_same_v<r, Tensor_t::node_type>);

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

   return 0;
}
