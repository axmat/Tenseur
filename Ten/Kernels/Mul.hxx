#ifndef TA_KERNELS_MUL_HXX
#define TA_KERNELS_MUL_HXX

#include <Ten/Types.hxx>

namespace ten::kernels {
// Multiply two vectors (elementwise)
template <class A, class B, class C>
   requires isDenseTensor<A>::value && isDenseTensor<B>::value &&
            isDenseTensor<C>::value
//&& same_element_type<A, B> && same_element_type<B, C> && same_size<A, B> &&
//same_size<B, C>
void mul(const A &a, const B &b, C &c) {
   using value_type = typename C::value_type;
   for (size_t i = 0; i < c.size(); i++)
      c[i] = static_cast<value_type>(a[i]) * static_cast<value_type>(b[i]);
}

// Multiply two dense matrices
template <class A, class B, class C> void mul(const A &a, const B &b, C &c) {
   size_t m = a.dim(0);
   size_t k = a.dim(1);
   size_t n = b.dim(1);
   using blas::transop;
   using T = typename A::value_type;
   const transop transa = (a.isTransposed() ? transop::trans : transop::no);
   const transop transb = (b.isTransposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : k);
   const size_t ldb = (transa == transop::no ? k : n);
   blas::gemm(transa, transb, m, n, k, T(1.), a.data(), lda, b.data(), ldb,
              T(0.), c.data(), m);
}

} // namespace ten::kernels

#endif
