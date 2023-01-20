/// \file Ten/Types.hxx

#ifndef TENSEUR_TEN_TYPES_HXX
#define TENSEUR_TEN_TYPES_HXX

#include <iostream>
#include <memory>
#include <type_traits>

namespace ten {
/// \enum format
/// Format type
enum class StorageFormat {
   /// Dense format
   Dense,
   /// Sparse coordinate format
   SparseCoo,
   /// Diagonal format
   Diagonal,
   /// Lower triangular format
   LowerTr,
   /// Upper triangular format
   UpperTr
};

#ifndef TENSEUR_SIZE_TYPE
#define TENSEUR_SIZE_TYPE std::size_t
#endif

/// \typedef size_type
/// Type of the indices
using size_type = TENSEUR_SIZE_TYPE;

// Forward declaration of shape
template <size_type Dim, size_type... Rest> class Shape;

/// \enum Order
/// Storage order of a multidimentional array
enum class StorageOrder { ColMajor, RowMajor };

static constexpr StorageOrder defaultOrder = StorageOrder::ColMajor;

/// \class tensor_base
/// Base class for tensor types
class TensorBase {
 public:
   virtual ~TensorBase() {}
};

// Expr is the base class of Tensor and Expressions (UnaryExpr and BinaryExpr)
template <class Derived> class Expr;

// Concept for Expression type
template <typename T>
concept isExpr = std::is_base_of_v<::ten::Expr<T>, T>;

// Scalar type
template <class T> class Scalar;

// Traits for scalar
template <class> struct isScalar : std::false_type {};
template <class T> struct isScalar<Scalar<T>> : std::true_type {};

// Forward declaration of tensor type
template <typename Scalar, typename Shape, StorageOrder order, typename Storage,
          typename Allocator>
class Tensor;

// Forward declaration of tensor operations
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct TensorOperations;

/*template<typename T>
using isTensor = std::is_same<T,
   Tensor<typename T::ScalarType, typename T::ShapeType,
      T::GetStorageOrder(), typename T::AllocatorType,
      typename T::StorageType, T::IsShort()>>;*/

template <typename> struct isTensor : std::false_type {};
template <typename Scalar, typename Shape, StorageOrder order, typename Storage,
          typename Allocator>
struct isTensor<Tensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = true;
};

// Dynamic tensor
template <typename> struct isDynamicTensor : std::false_type {};
template <typename Scalar, typename Shape, StorageOrder order, typename Storage,
          typename Allocator>
struct isDynamicTensor<Tensor<Scalar, Shape, order, Allocator, Storage>> {
   static constexpr bool value = Shape::isDynamic();
};

// Static tensor
template <typename> struct isStaticTensor : std::false_type {};
template <typename Scalar, typename Shape, StorageOrder order, typename Storage,
          typename Allocator>
struct isStaticTensor<Tensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::isStatic();
};

/// \typedef DefaultAllocator
/// Default allocator type
template <typename T> using DefaultAllocator = std::allocator<T>;

// Storage
template <class T, class Allocator> class DenseStorage;
template <class T, class Shape> class StaticDenseStorage;
// Storage traits
template <class> struct isDenseStorage : std::false_type {};
template <class T, class Allocator>
struct isDenseStorage<DenseStorage<T, Allocator>> : std::true_type {};
template <class T, class Allocator>
struct isDenseStorage<StaticDenseStorage<T, Allocator>> : std::true_type {};

// Storage of static shape
template <class> struct isStaticStorage : std::false_type {};
template <class T, class Allocator>
struct isStaticStorage<StaticDenseStorage<T, Allocator>> : std::true_type {};

/// \typedef DefaultStorage
/// Default storage type
template <class T, class Shape>
using DefaultStorage =
    std::conditional_t<Shape::isDynamic(), DenseStorage<T, DefaultAllocator<T>>,
                       StaticDenseStorage<T, Shape>>;

// Dense
template <class> struct isDenseTensor : std::false_type {};
template <typename Scalar, typename Shape, StorageOrder order, typename Storage,
          typename Allocator>
struct isDenseTensor<Tensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = isDenseStorage<Storage>::value;
};

////////////////////////////////////////////////////////////////////////////////
// Node types
// template<class> struct Node;

// Scalar node
template <class T> class ScalarNode;

// ScalarNode traits
template <class> struct isScalarNode : std::false_type {};
template <class T> struct isScalarNode<ScalarNode<T>> : std::true_type {};

// Tensor Node
template <class T, class Shape, StorageOrder Order, class Allocator,
          class Storage>
class TensorNode;

// TensorNode traits
template <class> struct isTensorNode : std::false_type {};
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct isTensorNode<TensorNode<T, Shape, Order, Storage, Allocator>>
    : std::true_type {};

// Unary Node
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class UnaryNode;

// Unary Expr
template <class E, template <typename...> class Func, typename... Args>
class UnaryExpr;

} // namespace ten

namespace ten::stack {
// Enable static tensor optimizations
// Allocate static tensor of size known at compile time on the stack
#ifndef TENSEUR_ENABLE_STACK_ALLOC
#define TENSEUR_ENABLE_STACK_ALLOC true
#endif
static constexpr bool enableStackAlloc = TENSEUR_ENABLE_STACK_ALLOC;

// TODO Set default maximum size of the stack in bytes
#ifndef TENSEUR_MAX_STACK_SIZE
#define TENSEUR_MAX_STACK_SIZE 400
#endif
static constexpr size_type maxStackSize = TENSEUR_MAX_STACK_SIZE;
} // namespace ten::stack

#endif
