# Tenseur
A header only C++20 tensor library [WIP]

## Features
- Multi dimensional arrays
- Support static, dynamic and mixed shape tensors
- Lazy evaluation of expressions
- BLAS backend for high performance numerical linear algebra
- Chain expressions

## Todo
- Operators precedence
- Check tensor shapes at compile time whenever possible
- Sparse tensors
- Special matrices
- Automatic differentiation
- Python binding

## Example
```
#include <Ten/Tensor>

using namespace ten;

int main() {
   auto a = iota<float, Shape<3, 3>>();
   auto b = iota<Tensor<float, Shape<3, 3>>>();
   auto c = fill<float, Shape<3>>(1.);

   auto e = a * b + c;
   auto x = e.eval();
}
```
