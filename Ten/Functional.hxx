#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <type_traits>

#include <Ten/Kernels/Host>
#include <Ten/Types.hxx>

namespace ten::functional {
////////////////////////////////////////////////////////////////////////////////
// Unary functions

/// Square root
template <class T, class R = T> struct Sqrt {
   using input_type = T;
   using output_type = R;

   static constexpr bool isParametric() { return false; }

   static constexpr void call(const T &a, R &b) {
      using value_type = typename R::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sqrt(static_cast<value_type>(a[i]));
      }
   }
};

/// Absolute value
template <class T, class R = T> struct Abs {
   using input_type = T;
   using output_type = R;

   static constexpr bool isParametric() { return false; }

   static constexpr void call(const T &a, R &b) {
      using value_type = typename R::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::abs(static_cast<value_type>(a[i]));
      }
   }
};

/// Power
template <class T, class R = T> struct Pow {
 private:
   double _n;

 public:
   using input_type = T;
   using output_type = R;

   explicit Pow(double n) : _n(n) {}

   static constexpr bool isParametric() { return true; }

   void call(const T &a, R &b) const {
      using value_type = typename R::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::pow(static_cast<value_type>(a[i]), _n);
      }
   }
};

template <class T, class R = typename T::scalarnode_type> struct Min {
   using input_type = T;
   using output_type = R;

   static constexpr bool isParametric() { return false; }

   static constexpr void call(const T &a, R &b) {
      using type = typename T::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::min(static_cast<type>(a[i]), res);
      }
      b = res;
   }
};

template <class T, class R = typename T::scalarnode_type> struct Max {
   using input_type = T;
   using output_type = R;

   static constexpr bool isParametric() { return false; }

   static constexpr void call(const T &a, R &b) {
      using type = typename T::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::max(static_cast<type>(a[i]), res);
      }
      b = res;
   }
};

////////////////////////////////////////////////////////////////////////////////
// Binary functions (Add, Sub, Mul and Div)
namespace details {
template <class, class> struct CommonType;

template <VectorNode A, VectorNode B>
   requires SameShape<A, B> && SameStorageOrder<A, B> && SameStorage<A, B> &&
            SameAllocator<A, B>
struct CommonType<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       TensorNode<value_type, typename A::shape_type, A::storageOrder(),
                  typename A::storage_type, typename A::allocator_type>;
};

template <class A, class B>
using common_type_t = typename CommonType<A, B>::type;
} // namespace details

// Binary operation
enum class BinaryOperation { add, sub, div };

// Binary function
template <BinaryOperation kind> struct BinaryFunc {

   template <class A, class B, class C = details::common_type_t<A, B>>
   struct Func {
      static_assert(A::isVector() && B::isVector(),
                    "Expected A and B to be vectors.");

      using left_input_type = A;
      using right_input_type = B;
      using output_type = C;

      using left_shape_type = typename A::shape_type;
      using right_shape_type = typename B::shape_type;
      using output_shape_type = typename C::shape_type;

      static constexpr output_shape_type
      outputShape(const left_shape_type &left, const right_shape_type &right) {
         output_shape_type s(left);
         return s;
      }

      static constexpr auto outputShape(const A &a, const B &b) {
         return a.shape();
      }

      static constexpr bool isParametric() { return false; }

      static constexpr void call(const A &left, const B &right, C &result) {
         size_t n = left.size();
         using value_type = typename C::value_type;
         for (size_t i = 0; i < n; i++) {
            switch (kind) {
            case BinaryOperation::add:
               result[i] = static_cast<value_type>(left[i]) +
                           static_cast<value_type>(right[i]);
               break;
            case BinaryOperation::sub:
               result[i] = static_cast<value_type>(left[i]) -
                           static_cast<value_type>(right[i]);
               break;
            case BinaryOperation::div:
               result[i] = static_cast<value_type>(left[i]) /
                           static_cast<value_type>(right[i]);
               break;
            }
         }
      }
   };
};

namespace details {
template <class, class> struct MulResult;

// vector * vector
template <VectorNode A, VectorNode B>
   requires SameShape<A, B> && SameStorageOrder<A, B> && SameStorage<A, B> &&
            SameAllocator<A, B>
struct MulResult<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       TensorNode<value_type, typename A::shape_type, A::storageOrder(),
                  typename A::storage_type, typename A::allocator_type>;
};

// matrix * matrix
template <MatrixNode A, MatrixNode B>
   requires SameStorageOrder<A, B> && SameStorage<A, B> && SameAllocator<A, B>
struct MulResult<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type = TensorNode<value_type,
                           Shape<A::shape_type::template staticDim<0>(),
                                 B::shape_type::template staticDim<1>()>,
                           A::storageOrder(), typename A::storage_type,
                           typename A::allocator_type>;
};

// matrix * vector
template <MatrixNode A, VectorNode B>
   requires SameStorageOrder<A, B> && SameStorage<A, B> && SameAllocator<A, B>
struct MulResult<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       TensorNode<value_type, Shape<A::shape_type::template staticDim<0>()>,
                  A::storageOrder(), typename A::storage_type,
                  typename A::allocator_type>;
};

// scalar * tensor
template <traits::ScalarNode A, traits::TensorNode B> struct MulResult<A, B> {
   using type = B;
};

} // namespace details

template <class A, class B, class C = typename details::MulResult<A, B>::type>
struct Mul;

// vector * vector
template <VectorNode A, VectorNode B, VectorNode C> struct Mul<A, B, C> {
   using left_input_type = A;
   using right_input_type = B;
   using output_type = C;

   using left_shape_type = typename A::shape_type;
   using right_shape_type = typename B::shape_type;
   using output_shape_type = typename C::shape_type;

   static constexpr bool isParametric() { return false; }

   static constexpr output_shape_type
   outputShape(const left_shape_type &left, const right_shape_type &right) {
      std::initializer_list<size_type> &&dims = {
          std::max(left.dim(0), right.dim(0))};
      output_shape_type s(std::move(dims));
      return s;
   }

   static constexpr void call(const A &left, const B &right, C &result) {
      size_t n = left.size();
      using value_type = typename C::value_type;
      for (size_t i = 0; i < n; i++) {
         result[i] = static_cast<value_type>(left[i]) *
                     static_cast<value_type>(right[i]);
      }
   }
};

// matrix * matrix
template <MatrixNode A, MatrixNode B, MatrixNode C> struct Mul<A, B, C> {
   using left_input_type = A;
   using right_input_type = B;
   using output_type = C;

   using left_shape_type = typename A::shape_type;
   using right_shape_type = typename B::shape_type;
   using output_shape_type = typename C::shape_type;

   static constexpr bool isParametric() { return false; }

   static constexpr output_shape_type
   outputShape(const left_shape_type &left, const right_shape_type &right) {
      std::initializer_list<size_type> &&dims = {left.dim(0), right.dim(1)};
      output_shape_type s(std::move(dims));
      return s;
   }

   static constexpr void call(const A &left, const B &right, C &result) {
      kernels::mul(left, right, result);
   }
};

// matrix * vector
template <MatrixNode A, VectorNode B, VectorNode C> struct Mul<A, B, C> {
   using left_input_type = A;
   using right_input_type = B;
   using output_type = C;

   using left_shape_type = typename A::shape_type;
   using right_shape_type = typename B::shape_type;
   using output_shape_type = typename C::shape_type;

   static constexpr bool isParametric() { return false; }

   static constexpr output_shape_type
   outputShape(const left_shape_type &left, const right_shape_type &right) {
      // FIXME transposed
      std::initializer_list<size_type> &&dims = {left.dim(0)};
      output_shape_type s(std::move(dims));
      return s;
   }

   static constexpr void call(const A &left, const B &right, C &result) {
      kernels::mul(left, right, result);
   }
};

// scalar * tensor
template <traits::ScalarNode A, traits::TensorNode B, traits::TensorNode C>
struct Mul<A, B, C> {
   using left_input_type = A;
   using right_input_type = B;
   using output_type = C;

   using right_shape_type = typename B::shape_type;
   using output_shape_type = typename C::shape_type;

   static constexpr bool isParametric() { return false; }

   static constexpr output_shape_type
   outputShape(const right_shape_type &right) {
      return right;
   }

   static constexpr void call(const A &left, const B &right, C &result) {
      size_t n = result.size();
      using value_type = typename C::value_type;
      for (size_t i = 0; i < n; i++) {
         result[i] = static_cast<value_type>(left.value()) *
                     static_cast<value_type>(right[i]);
      }
   }
};

} // namespace ten::functional

#endif
