#ifndef TA_KERNELS_STD_SIMD_ADD_HXX
#define TA_KERNELS_STD_SIMD_ADD_HXX

#include <experimental/bits/simd.h>
#include <experimental/simd>

#include <Ten/Config.hxx>
#include <Ten/Types.hxx>

namespace ten::kernels {

template <class A, class B, class C>
static void add(const A &a, const B &b, C &c) {
   size_t n = a.size();
   using T = typename A::value_type;
   constexpr size_t vlen = ::ten::simdVecLen;

   using vector_type = std::experimental::fixed_size_simd<float, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < n / vlen; i++) {
      // Load a and b
      size_t offset = i * vlen;
      vector_type a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // Sum
      vector_type c_vec = a_vec + b_vec;
      // Copy back
      c_vec.copy_to(c.data() + offset, alignment{});
   }
   for (size_t i = vlen * (n / vlen); i < n; i++) {
      c[i] = a[i] + b[i];
   }
}
} // namespace ten::kernels

#endif
