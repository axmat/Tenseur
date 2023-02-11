[![docs](https://readthedocs.org/projects/tenseur/badge/?version=latest)](https://tenseur.readthedocs.io/en/latest/index.html)

## Tenseur
A header only C++20 tensor library [WIP]

### Features
- Multi dimensional arrays
- Support static, dynamic and mixed shape tensors
- Lazy evaluation of expressions
- BLAS backend for high performance numerical linear algebra
- Chain expressions
- Factory functions: fill, ones, zeros, iota, rand

### Todo
- Compile to a shared library
- Tests for shared library
- Basic python bindings
- Tests
- CI/CD
- Check tensor shapes at compile time whenever possible
- Sparse tensors
- Special matrices
- Automatic differentiation
- Python documentation
- C++ API documentation
- Untyped tensor
- Operators precedence
- Inplace operations

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

## Build the examples
```
mkdir build-examples
cd build-examples
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_EXAMPLES=ON
cmake --build . --
```

## Build the tests
```
mkdir build-tests
cd build-tests
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_TESTS=ON
cmake --build . --
```

## Build the docs
```
mkdir build-docs
cd build-docs
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_DOCS=ON
cmake --build . --
```

