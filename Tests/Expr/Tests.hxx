#ifndef TENSEUR_TESTS_EXPR_TESTS
#define TENSEUR_TESTS_EXPR_TESTS

#include <Ten/Types.hxx>
#include <gtest/gtest.h>
#include <type_traits>

namespace ten::tests {

using testing::AssertionFailure;
using testing::AssertionResult;
using testing::AssertionSuccess;

// Compare the value type of two tensors
template <class A, class B>
testing::AssertionResult same_value_type(const A &a, const B &b) {
   using T = typename A::value_type;
   using R = typename B::value_type;
   if (!std::is_same_v<T, R>)
      return AssertionFailure()
             << "Different value type " << ::ten::to_string<T>() << " and "
             << ::ten::to_string<R>() << std::endl;
   return testing::AssertionSuccess();
}

// Compare the shape of two tensors
template <class A, class B>
testing::AssertionResult same_shape(const A &a, const B &b) {
   if (A::rank() != B::rank())
      return AssertionFailure() << "Different rank " << A::rank() << " and "
                                << B::rank() << std::endl;

   if (a.size() != b.size())
      return AssertionFailure() << "Different sizes " << a.size() << " and "
                                << b.size() << std::endl;

   for (size_t i = 0; i < A::rank(); i++)
      if (a.dim(i) != b.dim(i))
         return AssertionFailure()
                << "Different dimensions at index " << i << std::endl;
   return testing::AssertionSuccess();
}

// Compare the values of two floating point tensors
template <class A, class B>
   requires std::is_floating_point_v<typename A::value_type> &&
            std::is_floating_point_v<typename B::value_type>
testing::AssertionResult same_values(const A &a, const B &b,
                                      const double tol = 1e-3) {
   for (size_t i = 0; i < a.size(); i++) {
      auto val = std::abs(a[i] - b[i]);
      if (static_cast<double>(val) > tol)
         return testing::AssertionFailure()
                << "Different values at index " << i;
   }
   return testing::AssertionSuccess();
}

// Compare the values of two integral tensors
template <class A, class B>
   requires std::is_integral_v<typename A::value_type> &&
            std::is_integral_v<typename B::value_type>
testing::AssertionResult same_values(const A &a, const B &b) {
   for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i])
         return testing::AssertionFailure()
                << "Different values at index " << i;
   }
   return testing::AssertionSuccess();
}

// Compare two tensors
template <class A, class B>
AssertionResult equal(const A &a, const B &b, double eps = 1e-3) {
   AssertionResult r_type = same_value_type(a, b);
   if (!r_type)
      return r_type;
   AssertionResult r_shape = same_shape(a, b);
   if (!r_shape)
      return r_shape;
   AssertionResult r_values = same_values(a, b, eps);
   if (!r_values)
      return r_values;

   return AssertionSuccess();
}

} // namespace ten::tests

#endif
