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
   auto a = iota<Matrix<float>>({3, 3});
   auto b = iota<Matrix<float>>({3, 3});
   auto c = ones<Vector<float>>(3);

   Vector<float> x = a * b + c;
}
```
