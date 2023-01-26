#include <ios>
#include <iostream>

#include <Ten/Tensor>
#include <memory>
#include <type_traits>

int main() {
   using namespace ten;
   using namespace std;

   {
      Shape<dynamic, 3, 4> s({2, 3, 4});

      std::cout << "Number of dims\n";
      std::cout << s.rank() << std::endl;

      std::cout << "Dynamic shape at 0\n";
      std::cout << s.isDynamicDim<0>() << std::endl;
      cout << s.staticDim<0>() << endl;

      cout << "Static shape at 1\n";
      cout << s.isDynamicDim<1>() << endl;
      cout << s.dim(1) << endl;
      std::cout << "Dynamic shape at 0\n";
   }

   {
      cout << "Shape of static dim 2 x 3" << endl;
      Shape<2, 3> a;
      cout << a.staticDim<0>() << endl;
      cout << a.staticDim<1>() << endl;
      cout << "Is dynamic = " << std::boolalpha << a.isDynamic() << endl;
      cout << a.dim(0) << endl;
      cout << a.dim(1) << endl;
   }

   {
      cout << "\nDyanmic or static shape\n";
      cout << "s\n";
      Shape<dynamic, 2> s({2, 2});
      cout << s.isDynamic() << endl;
      cout << s.isStatic() << endl;

      cout << "a\n";
      Shape<3, 3, 5, 2, dynamic> a({3, 3, 5, 2, 4});
      cout << a.isDynamic() << endl;
      cout << a.isStatic() << endl;

      cout << "b\n";
      Shape<3, 3, 5, 2> b;
      cout << b.isDynamic() << endl;
      cout << b.isStatic() << endl;
   }

   {
      cout << "\nStorage\n";
      DynamicShape<2> shape({2, 3});
      DenseStorage<float, std::allocator<float>> s(shape.size());
      for (size_t i = 0; i < 6; i++)
         s[i] = i;
      for (size_t i = 0; i < 6; i++)
         cout << s[i] << "\n";
   }

   {
      cout << "\nTensor with static storage\n";
      Tensor<float, Shape<2, 3>> a;
      using tensor_type = decltype(a);
      static_assert(std::is_same_v<tensor_type::storage_type,
                                   StaticDenseStorage<float, Shape<2, 3>>>);
      cout << "Rank = " << a.rank() << endl;
      cout << "Order = "
           << (a.storageOrder() == StorageOrder::RowMajor ? "RowMajor"
                                                          : "ColMajor")
           << endl;
      cout << "IsVector = " << a.isVector() << endl;
      cout << "IsMatrix = " << a.isMatrix() << endl;
      cout << "IsTransposed = " << a.isTransposed() << endl;
   }

   {
      using namespace ten::details;

      using A = MakeDynamicSequence<1>;
      static_assert(std::is_same_v<A::type::shape_type, Shape<dynamic>>);

      using dA = DynamicShape<1>;
      static_assert(std::is_same_v<dA, Shape<dynamic>>);

      using B = MakeDynamicSequence<2>;
      static_assert(
          std::is_same_v<B::type::shape_type, Shape<dynamic, dynamic>>);

      using dB = DynamicShape<2>;
      static_assert(std::is_same_v<dB, Shape<dynamic, dynamic>>);

      using C = MakeDynamicSequence<3>;
      static_assert(std::is_same_v<C::type::shape_type,
                                   Shape<dynamic, dynamic, dynamic>>);

      using dC = DynamicShape<3>;
      static_assert(std::is_same_v<dC, Shape<dynamic, dynamic, dynamic>>);
   }

   {
      cout << "\nTensor shape and static strides\n";
      Shape<2, 3, 4> s;
      using S = decltype(s);

      using Stride_t = Stride<S, StorageOrder::ColMajor>;
      cout << "Stride_0 = " << Stride_t::template staticDim<0>() << endl;
      cout << "Stride_1 = " << Stride_t::template staticDim<1>() << endl;
      cout << "Stride_2 = " << Stride_t::template staticDim<2>() << endl;

      cout << S::staticDim<0>() << "x" << S::staticDim<1>() << "x"
           << S::staticDim<2>() << endl;
      cout << details::NthStaticStride<0, S>::value << endl;
      cout << details::NthStaticStride<1, S>::value << endl;
      cout << details::NthStaticStride<2, S>::value << endl;

      auto v = details::computeStaticStrides<S, StorageOrder::ColMajor>();
      cout << "v = {";
      cout << v[0] << ", " << v[1] << ", " << v[2] << "}" << endl;
   }

   {
      cout << "\nDynamic strides\n";
      DynamicShape<3> shape({2, 3, 4});
      Stride<DynamicShape<3>, StorageOrder::ColMajor> stride(shape);
      for (size_t i = 0; i < 3; i++)
         cout << stride.dim(i) << endl;
   }

   {
      cout << "\nTensor of dynamic shape\n";
      Tensor<float, Shape<dynamic, dynamic>> A({3, 4});
      using Tensor_t = decltype(A);
      using Shape_t = Tensor_t::shape_type;
      const Tensor_t::shape_type &shape = A.shape();

      cout << "Rank = " << Shape_t::rank() << endl;
      cout << "Dynamic shape = [";
      cout << shape.dim(0) << ", ";
      cout << shape.dim(1) << "]\n";
      cout << "Is dynamic dims = \n";
      cout << Shape_t::isDynamicDim<0>() << endl;
      cout << Shape_t::isDynamicDim<1>() << endl;
      cout << "Dynamic size = " << shape.size() << endl;
      cout << "Must be uninit\n";
      using Node_t = Tensor_t::node_type;
      std::weak_ptr<Node_t> s = A.node();
      cout << A.node().get() << endl;
      auto p = s.lock();
      cout << p.get() << endl;
   }

   {
      cout << "Acess elements" << endl;
      DynamicTensor<float, 1> a({4});
      auto n = a.node();
      for (size_t i = 0; i < 4; i++)
         a[i] = i;
      for (size_t i = 0; i < 4; i++)
         cout << a[i] << endl;
   }

   {
      cout << "\nStatic Tensor - uninitialized memory\n";
      Tensor<float, Shape<10>> x;
      float *data = x.data();
      for (size_t i = 0; i < 10; i++)
         data[i] = float(i);
      for (size_t i = 0; i < 10; i++) {
         cout << "x[" << i << "] = " << data[i] << endl;
      }
   }

   {
      cout << "\nStatic Matrix\n";
      Tensor<float, Shape<2, 3>> x;
      float *data = x.data();
      for (size_t i = 0; i < 6; i++)
         data[i] = float(i);
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            cout << "x(" << i << "," << j << ") = " << x(i, j) << endl;
         }
      }
   }

   {
      cout << "fill static tensor" << endl;
      auto x = fill<Tensor<float, Shape<10>>>(3.);
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = fill<float, Shape<10>>(3.);
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      cout << "fill dynamic tensor" << endl;
      auto x = fill<Tensor<float, Shape<2, dynamic>>>(
         Shape<2, dynamic>({2, 3}), 3.);
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = fill<Tensor<float, Shape<2, dynamic>>>({2, 3}, 3.);
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = fill<float, Shape<2, dynamic>>(Shape<2, dynamic>({2, 3}),
         3.);
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;

      auto t = fill<float, Shape<2, dynamic>>({2, 3}, 3.);
      cout << "t = ";
      for (size_t i = 0; i < 6; i++) {
         cout << t[i] << " ";
      }
      cout << endl;
   }


   {
      cout << "zeros static tensor" << endl;
      auto x = zeros<Tensor<float, Shape<10>>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = zeros<float, Shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      cout << "zeros dynamic tensor" << endl;
      auto x = zeros<Tensor<float, Shape<2, dynamic>>>(
         Shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = zeros<Tensor<float, Shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = zeros<float, Shape<2, dynamic>>(Shape<2, dynamic>({2, 3}));
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;

      auto t = zeros<float, Shape<2, dynamic>>({2, 3});
      cout << "t = ";
      for (size_t i = 0; i < 6; i++) {
         cout << t[i] << " ";
      }
      cout << endl;
   }

   {
      cout << "ones static tensor" << endl;
      auto x = ones<Tensor<float, Shape<10>>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = ones<float, Shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;
   }

   {
      cout << "ones dynamic tensor" << endl;
      auto x = ones<Tensor<float, Shape<2, dynamic>>>(
         Shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = ones<Tensor<float, Shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = ones<float, Shape<2, dynamic>>(Shape<2, dynamic>({2, 3}));
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;

      auto t = ones<float, Shape<2, dynamic>>({2, 3});
      cout << "t = ";
      for (size_t i = 0; i < 6; i++) {
         cout << t[i] << " ";
      }
      cout << endl;
   }

   {
      cout << "iota static tensor" << endl;
      auto x = iota<Tensor<float, Shape<10>>>();
      cout << "x = ";
      for (size_t i = 0; i < 10; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = iota<float, Shape<10>>();
      cout << "y = ";
      for (size_t i = 0; i < 10; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

   }

   {
      cout << "iota dynamic tensor" << endl;
      auto x = iota<Tensor<float, Shape<2, dynamic>>>(
         Shape<2, dynamic>({2, 3}));
      cout << "x = ";
      for (size_t i = 0; i < 6; i++) {
         cout << x[i] << " ";
      }
      cout << endl;

      auto y = iota<Tensor<float, Shape<2, dynamic>>>({2, 3});
      cout << "y = ";
      for (size_t i = 0; i < 6; i++) {
         cout << y[i] << " ";
      }
      cout << endl;

      auto z = iota<float, Shape<2, dynamic>>(Shape<2, dynamic>({2, 3}));
      cout << "z = ";
      for (size_t i = 0; i < 6; i++) {
         cout << z[i] << " ";
      }
      cout << endl;

      auto t = iota<float, Shape<2, dynamic>>({2, 3});
      cout << "t = ";
      for (size_t i = 0; i < 6; i++) {
         cout << t[i] << " ";
      }
      cout << endl;
   }


   return 0;
}
