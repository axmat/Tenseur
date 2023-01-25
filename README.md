# Tenseur
A header only C++20 tensor library [WIP]

## Features
- Multi dimensional arrays
- Support static, dynamic and mixed shape tensors
- Lazy evaluation of expressions
- BLAS backend for high performance numerical linear algebra

## Todo
- Chain expressions
- Check tensor shapes at compile time whenever possible
- Sparse tensors
- Special matrices
- Automatic differentiation
- Python binding

## Example
```
DynamicTensor<float, 1> a({3});
DynamicTensor<float, 1> b({3});
for (size_t i = 0; i < 3; i++) {
   a[i] = i;
   b[i] = i;
}

auto c = (a + b).eval();
```
