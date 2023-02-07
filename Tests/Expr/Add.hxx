#ifndef TENSEUR_TESTS_EXPR_ADD
#define TENSEUR_TESTS_EXPR_ADD

#include <Ten/Tensor.hxx>

#include "Ref.hxx"
#include "Tests.hxx"

using namespace ten;

TEST(Add, DenseVector_DenseVector) {
   size_t size = 10;
   Vector<float> a(size);
   Vector<float> b(size);
   auto c_ref = ten::tests::add(a, b);
   auto c = (a + b).eval();

   ASSERT_TRUE(tests::equal(c, c_ref));
}

#endif
