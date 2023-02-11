#ifndef TENSEUR_TESTS_EXPR_REF
#define TENSEUR_TESTS_EXPR_REF

#include <Ten/Types.hxx>

namespace ten::tests {
// Dynamic tensors
template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
auto add(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Same shape");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
auto mul(const A &a, const B &b) {
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

// Static tensors
template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value &&
            (A::size() == B::size())
auto adds(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value &&
            (A::size() == B::size())
auto muls(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

} // namespace ten::tests

#endif
