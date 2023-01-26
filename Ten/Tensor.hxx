/// \file Ten/Tensor.hxx

#ifndef TENSEUR_TENSOR_HXX
#define TENSEUR_TENSOR_HXX

#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Ten/Expr.hxx>
#include <Ten/Functional.hxx>
#include <Ten/Shape.hxx>
#include <Ten/Types.hxx>
#include <Ten/Utils.hxx>

#include <Ten/Storage/DenseStorage.hxx>

namespace ten {

/// \class Expr
/// Represent an expression
template <typename Derived> class Expr {
 public:
   // using type = static_cast<Derived>(*this);

 protected:
   Expr() = default;
   Expr(const Expr &) = default;
   Expr(Expr &&) = default;
};

// Add two expr
template <typename LeftExpr, typename RightExpr>
   requires ::ten::isExpr<std::remove_cvref_t<LeftExpr>> &&
            ::ten::isExpr<std::remove_cvref_t<RightExpr>>
auto operator+(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::BinaryExpr<typename L::node_type, typename R::node_type,
                            ::ten::functional::Add>(left.node(), right.node());
}

template <typename T, typename E>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator+(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return Scalar<T>(scalar) + std::forward<R>(expr);
}

template <typename E, typename T>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator+(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) + Scalar<T>(scalar);
}

// Substract two expressions
template <typename LeftExpr, typename RightExpr>
   requires ::ten::isExpr<std::remove_cvref_t<LeftExpr>> &&
            ::ten::isExpr<std::remove_cvref_t<RightExpr>>
auto operator-(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::BinaryExpr<typename L::node_type, typename R::node_type,
                            ::ten::functional::Sub>(left.node(), right.node());
}

template <typename T, typename E>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator-(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return Scalar<T>(scalar) - std::forward<R>(expr);
}

template <typename E, typename T>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator-(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) - Scalar<T>(scalar);
}

// Multiply two expressions
template <typename LeftExpr, typename RightExpr>
   requires ::ten::isExpr<std::remove_cvref_t<LeftExpr>> &&
            ::ten::isExpr<std::remove_cvref_t<RightExpr>>
auto operator*(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::BinaryExpr<typename L::node_type, typename R::node_type,
                            ::ten::functional::Mul>(left.node(), right.node());
}

template <typename T, typename E>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator*(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return Scalar<T>(scalar) * std::forward<R>(expr);
}

template <typename E, typename T>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator*(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) * Scalar<T>(scalar);
}

// Divide two expressions
template <typename LeftExpr, typename RightExpr>
   requires ::ten::isExpr<std::remove_cvref_t<LeftExpr>> &&
            ::ten::isExpr<std::remove_cvref_t<RightExpr>>
auto operator/(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::BinaryExpr<typename L::node_type, typename R::node_type,
                            ::ten::functional::Div>(left.node(), right.node());
}

template <typename T, typename E>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator/(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return Scalar<T>(scalar) / std::forward<R>(expr);
}

template <typename E, typename T>
   requires ::ten::isExpr<std::remove_cvref_t<E>>
auto operator/(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) / Scalar<T>(scalar);
}

/// \class ScalarOperations
template <class T> struct ScalarOperations {
   [[nodiscard]] inline static constexpr size_type rank() { return 0; }
};

/// \class ScalarNode
/// Node of scalar type
template <typename T>
class ScalarNode : public ScalarOperations<ScalarNode<T>> {
 public:
   using scalar_type = Scalar<T>;
   // for std::conditional_t
   using tensor_type = void;

 private:
   T _value;

 public:
   ScalarNode() : _value(T()) {}

   explicit ScalarNode(const T &value) : _value(value) {}
   explicit ScalarNode(T &&value) : _value(std::move(value)) {}

   const T &value() const { return _value; }

   ScalarNode &operator=(const T &value) {
      _value = value;
      return *this;
   }
};

/// \class Scalar
/// Hold a single value of type T.
template <typename T>
class Scalar : public Expr<Scalar<T>>, public ScalarOperations<Scalar<T>> {
 public:
   using node_type = ScalarNode<T>;

 private:
   std::shared_ptr<node_type> _node;

 public:
   explicit Scalar(const T &value)
       : _node(std::make_shared<node_type>(value)) {}
   explicit Scalar(T &&value)
       : _node(std::make_shared<node_type>(std::move(value))) {}

   explicit Scalar(std::shared_ptr<node_type> node) : _node(node) {}

   const T &value() const { return _node.get()->value(); }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }
};

/// \class TensorOperations
/// Tensor operations
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct TensorOperations {
   /// \typedef value_type
   /// Value type
   using value_type = T;

   /// \typedef scalar_type
   /// Scalar type
   using scalar_type = Scalar<T>;

   /// \typedef shape_type
   /// Shape type
   using shape_type = Shape;

   /// \typedef storage_type
   /// Storage type
   using storage_type = Storage;

   /// \typedef allocator_type
   /// Type of the allocator
   using allocator_type = Allocator;

   /// \typedef stride_type
   /// Stride type
   using stride_type = Stride<Shape, Order>;

   /// Returns the storage order
   [[nodiscard]] inline static constexpr StorageOrder storageOrder() {
      return Order;
   }

   //// Returns the rank
   [[nodiscard]] inline static constexpr size_type rank() {
      return Shape::rank();
   }

   /// Returns the static size
   [[nodiscard]] inline static constexpr size_type staticSize()
      requires(Shape::isStatic())
   {
      return Shape::staticSize();
   }

   /// Returns whether the tensor is of static shape
   [[nodiscard]] inline static constexpr bool isStatic() {
      return Shape::isStatic();
   }

   /// Returns whether the tensor is of dynamic shape
   [[nodiscard]] inline static constexpr bool isDynamic() {
      return Shape::isDynamic();
   }

   /// Returns whether the Index'th dim is static
   template <size_type Index>
   [[nodiscard]] inline static constexpr bool isStaticDim() {
      return Shape::template isStaticDim<Index>();
   }

   /// Returns whether the Index'th dim is dynamic
   template <size_type Index>
   [[nodiscard]] inline static constexpr bool isDynamicDim() {
      return Shape::template isDynamicDim<Index>();
   }

   /// Return the Index'th static dimension
   template <size_type Index>
   [[nodiscard]] inline static constexpr size_type staticDim() {
      return Shape::template staticDim<Index>();
   }

   /// Returns whether the tensor is a vector
   [[nodiscard]] static constexpr bool isVector() { return Shape::rank() == 1; }

   /// Returns whether the tensor is a matrix
   [[nodiscard]] static constexpr bool isMatrix() { return Shape::rank() == 2; }
};

/// \class TensorNode
/// Tensor node
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
class TensorNode
    : public TensorOperations<T, Shape, Order, Storage, Allocator> {
 public:
   /// \typedef base_type
   /// Base type
   using base_type = TensorOperations<T, Shape, Order, Allocator, Storage>;

   /// Tensor type
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;

   using scalarnode_type = ScalarNode<T>;

   using stride_type = typename base_type::stride_type;

 private:
   /// Optional shape (only for dynamic tensors)
   std::optional<Shape> _shape;
   /// optional stride (only for dynamic tensors)
   std::optional<stride_type> _stride;
   /// Storage
   Storage _storage;
   /// Is transposed
   bool _transposed = false;

 private:
   /// Returns the value at the indices
   [[nodiscard]] inline typename base_type::value_type &
   at(size_type index, auto... tail) noexcept {
      static constexpr size_type rank = Shape::rank();
      constexpr size_type n = sizeof...(tail);
      static_assert(n == 0 || n == (rank - 1), "Invalid number of indices.");
      if constexpr (rank == 1) {
         return _storage[index];
      }
      std::array<size_type, Shape::rank()> indices{
          index, static_cast<size_type>(tail)...};
      if constexpr (Shape::isDynamic()) {
         size_type idx = details::linearIndex(_stride.value(), indices);
         return _storage[idx];
      } else {
         size_type idx = details::staticLinearIndex<stride_type>(indices);
         return _storage[idx];
      }
   }

   /// Returns the value at the indices
   [[nodiscard]] inline const typename base_type::value_type &
   at(size_type index, auto... tail) const noexcept {
      static constexpr size_type rank = Shape::rank();
      constexpr size_type n = sizeof...(tail);
      static_assert(n == 0 || n == (rank - 1), "Invalid number of indices.");
      if constexpr (rank == 1) {
         return _storage[index];
      }
      std::array<size_type, Shape::rank()> indices{
          index, static_cast<size_type>(tail)...};
      if constexpr (Shape::isDynamic()) {
         size_type idx = details::linearIndex(_stride.value(), indices);
         return _storage[idx];
      } else {
         size_type idx = details::staticLinearIndex<stride_type>(indices);
         return _storage[idx];
      }
   }

 public:
   /// Construct a static TensorNode
   TensorNode() noexcept
      requires(Shape::isStatic())
       : _shape(std::nullopt), _stride(std::nullopt), _storage(Storage()) {}

   /// Construct a TensorNode from a list of shape
   explicit TensorNode(std::initializer_list<size_type> &&shape) noexcept
      requires(Shape::isDynamic())
       : _shape(std::move(shape)), _storage(Storage(_shape.value().size())),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Construct a TensorNode from the shape
   explicit TensorNode(const Shape &shape) noexcept
       : _shape(shape), _storage(shape.size()),
         _stride(typename base_type::stride_type(_shape.value())) {}

   [[nodiscard]] size_type dim(size_type index) const {
      return _shape.value().dim(index);
   }

   [[nodiscard]] size_type size() const {
      if constexpr (Shape::isDynamic()) {
         return _shape.value().size();
      } else {
         return Shape::staticSize();
      }
   }

   [[nodiscard]] const Shape &shape() const { return _shape.value(); }

   [[nodiscard]] T *data() { return _storage.data(); }
   [[nodiscard]] const T *data() const { return _storage.data(); }

   [[nodiscard]] bool isTransposed() const { return _transposed; }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      return at(index);
   }
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      return at(index);
   }

   /// Overloading the () operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator()(auto... index) const noexcept {
      return at(index...);
   }
   [[nodiscard]] inline typename base_type::value_type &
   operator()(auto... index) noexcept {
      return at(index...);
   }
};

namespace details {
template <typename Storage> struct AllocatorType {
   using type = typename Storage::allocator_type;
};
template <class T, class Shape>
struct AllocatorType<::ten::StaticDenseStorage<T, Shape>> {
   using type = void;
};
} // namespace details

/// \class Tensor
///
/// Tensor represented by a multidimentional array.
template <typename T, typename Shape, StorageOrder Order = defaultOrder,
          typename Storage = DefaultStorage<T, Shape>,
          typename Allocator = typename details::AllocatorType<Storage>::type>
class Tensor final
    : public Expr<Tensor<T, Shape, Order, Storage, Allocator>>,
      public TensorOperations<T, Shape, Order, Storage, Allocator> {
 public:
   /// \typedef scalar_type
   /// Scalar type
   using scalar_type = Scalar<T>;

   // TODO Type of the casted tensor
   // T must be convertible to To
   /*
  template <typename To>
  requires std::convertible_to<T, To>
  using casted_type = Tensor<T, Shape, Order,
      Storage::template casted_type<To>,
      Allocator::template casted_type<To>>;*/

   /// \typedef base_type
   /// Type of the tensor operations.
   using base_type = TensorOperations<T, Shape, Order, Storage, Allocator>;

   /// \typedef node_type
   /// Node type
   using node_type = TensorNode<T, Shape, Order, Storage, Allocator>;

 private:
   /// Shared pointer to the node
   // TODO Slice / View / or TilledTensor
   std::shared_ptr<node_type> _node;

 public:
   /// Constructor for static Tensor
   Tensor() noexcept
      requires(Shape::isStatic())
       : _node(std::make_shared<node_type>()) {}

   /// Constructor for Tensor with a storage of type Ten::DenseStorage
   explicit Tensor(std::initializer_list<size_type> &&shape) noexcept
      requires(Shape::isDynamic())
       : _node(std::make_shared<node_type>(std::move(shape))) {}

   /// Constructor of Tensor from shape
   explicit Tensor(const Shape &shape) noexcept
       : _node(std::make_shared<node_type>(shape)) {}

   /// Constructor of Tensor from a shared pointer to TensorNode
   Tensor(const std::shared_ptr<node_type> &node) : _node(node) {}

   /// Assignment operator
   Tensor(const Tensor &t) { _node = t._node; }

   Tensor(Tensor &&) = default;

   Tensor &operator=(const Tensor &t) {
      _node = t._node;
      return *this;
   }

   Tensor &operator=(Tensor &&t) = default;

   // TODO Iterators

   /// Returns the shape
   [[nodiscard]] inline const Shape &shape() const {
      return _node.get()->shape();
   }

   /// Returns the dynamic size
   [[nodiscard]] inline size_type size() const { return _node.get()->size(); }

   /// Returns the index'th dynamic dimension
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _node.get()->dim(index);
   }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Get the data
   [[nodiscard]] const T *data() const { return _node.get()->data(); }
   [[nodiscard]] T *data() { return _node.get()->data(); }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      return (*_node.get())[index];
   }
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      return (*_node.get())[index];
   }

   /// Overloading the () operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator()(auto... index) const noexcept {
      return (*_node.get())(index...);
   }
   [[nodiscard]] inline typename base_type::value_type &
   operator()(auto... index) noexcept {
      return (*_node.get())(index...);
   }

   /// Is transposed
   [[nodiscard]] bool isTransposed() const {
      return _node.get()->isTransposed();
   }
};

/// \typedef DynamicTensor
/// Dynamic tensor
template <typename T, size_type Rank, StorageOrder order = defaultOrder,
          typename Storage = DefaultStorage<T, DynamicShape<Rank>>,
          typename Allocator = typename Storage::allocator_type>
using DynamicTensor = Tensor<T, DynamicShape<Rank>, order, Storage, Allocator>;

// Static tensor
// FIXME Order?
// template <typename T, size_type...dims>
// using StaticTensor = Tensor<T, Shape<dims...>>;

////////////////////////////////////////////////////////////////////////////////
// Functions for creating a new tensor

// fill<Tensor<...>>(value)
template <class T>
   requires(::ten::isDenseTensor<T>::value)
[[nodiscard]] auto fill(typename T::value_type value) {
   T x;
   for (size_type i = 0; i < x.size(); i++)
      x[i] = value;
   return x;
}

// fill<T, Shape, Order, Storage, Allocator>(value)
template <class T, class Shape, StorageOrder Order = defaultOrder,
          class Storage = DefaultStorage<T, Shape>,
          class Allocator = typename details::AllocatorType<Storage>::type>
   requires(
       ::ten::isDenseTensor<Tensor<T, Shape, Order, Storage, Allocator>>::value)
[[nodiscard]] auto fill(T value) {
   return fill<Tensor<T, Shape, Order, Storage, Allocator>>(value);
}

// fill<Tensor<...>>(shape, value)
template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto fill(typename T::shape_type &&shape,
                        typename T::value_type value) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   T x(std::forward<shape_type>(shape));
   for (size_type i = 0; i < x.size(); i++) {
      x[i] = value;
   }
   return x;
}

template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims,
                        typename T::value_type value) {
   using shape_type = typename T::shape_type;
   return fill<T>(shape_type(std::move(dims)), value);
}

// fill<T, shape, Order, Storage, Allocator>(shape, value)
template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto fill(Shape &&shape, T value) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   return fill<tensor_type>(std::forward<Shape>(shape), value);
}

template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims, T value) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return fill<tensor_type>(shape_type(std::move(dims)), value);
}

// zeros<Tensor<...>>()
template <class T>
   requires(::ten::isStaticTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto zeros() {
   using value_type = typename T::value_type;
   return fill<T>(value_type(0));
}

// zeros<T, Shape, Order, Storage, Allocator>()
template <class T, class Shape, StorageOrder Order = defaultOrder,
          class Storage = DefaultStorage<T, Shape>,
          class Allocator = typename details::AllocatorType<Storage>::type>
   requires(::ten::isStaticTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto zeros() {
   return zeros<Tensor<T, Shape, Order, Storage, Allocator>>();
}

// zeros<Tensor<...>>(shape)
template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto zeros(typename T::shape_type &&shape) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   return fill<T>(std::forward<shape_type>(shape), value_type(0));
}

template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using shape_type = typename T::shape_type;
   return zeros<T>(shape_type(std::move(dims)));
}

// zeros<T, Shape, Order, Storage, Allocator>(shape)
template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto zeros(Shape &&shape) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   return zeros<tensor_type>(std::forward<Shape>(shape));
}

template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return zeros<tensor_type>(shape_type(std::move(dims)));
}

// ones<Tensor<...>>()
template <class T>
   requires(::ten::isStaticTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto ones() {
   using value_type = typename T::value_type;
   return fill<T>(value_type(1));
}

// ones<T, Shape, Order, Storage, Allocator>()
template <class T, class Shape, StorageOrder Order = defaultOrder,
          class Storage = DefaultStorage<T, Shape>,
          class Allocator = typename details::AllocatorType<Storage>::type>
   requires(::ten::isStaticTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto ones() {
   return ones<Tensor<T, Shape, Order, Storage, Allocator>>();
}

// ones<Tensor<...>>(shape)
template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto ones(typename T::shape_type &&shape) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   return fill<T>(std::forward<shape_type>(shape), value_type(1));
}

template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using shape_type = typename T::shape_type;
   return ones<T>(shape_type(std::move(dims)));
}

// ones<T, Shape, Order, Storage, Allocator>(shape)
template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto ones(Shape &&shape) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   return ones<tensor_type>(std::forward<Shape>(shape));
}

template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return ones<tensor_type>(shape_type(std::move(dims)));
}

// iota<Tensor<...>>(value)
template <class T>
   requires(::ten::isStaticTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto
iota(typename T::value_type value = typename T::value_type(0)) {
   using value_type = typename T::value_type;
   T x;
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}

template <class T, class Shape, StorageOrder Order = defaultOrder,
          class Storage = DefaultStorage<T, Shape>,
          class Allocator = typename details::AllocatorType<Storage>::type>
   requires(::ten::isStaticTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto iota(T value = T(0)) {
   return iota<Tensor<T, Shape, Order, Storage, Allocator>>(value);
}

// iota<Tensor<...>>(shape, value)
template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto
iota(typename T::shape_type &&shape,
     typename T::value_type value = typename T::value_type(0)) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   T x(std::forward<shape_type>(shape));
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}
template <class T>
   requires(::ten::isDynamicTensor<T>::value &&
            ::ten::isDenseStorage<typename T::storage_type>::value)
[[nodiscard]] auto
iota(std::initializer_list<size_type> &&dims,
     typename T::value_type value = typename T::value_type(0)) {
   using shape_type = typename T::shape_type;
   return iota<T>(shape_type(std::move(dims)), value);
}

// iota<T, Shape, Order, Storage, Allocator>(shape, value)
template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto iota(Shape &&shape, T value = T(0)) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   return iota<tensor_type>(std::forward<Shape>(shape), value);
}

template <
    class T, class Shape, StorageOrder Order = defaultOrder,
    class Storage = ::ten::DefaultStorage<T, Shape>,
    class Allocator = typename ::ten::details::AllocatorType<Storage>::type>
   requires(::ten::isDynamicTensor<
                Tensor<T, Shape, Order, Storage, Allocator>>::value &&
            ::ten::isDenseStorage<Storage>::value)
[[nodiscard]] auto iota(std::initializer_list<size_type> &&dims,
                        T value = T(0)) {
   using tensor_type = Tensor<T, Shape, Order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return iota<tensor_type>(shape_type(std::move(dims)), value);
}

// fill, ones, zeros and iota for default float tensors
template <class T, size_type... dims>
[[nodiscard]] auto fill(T value) -> Tensor<T, Shape<dims...>> {
   using tensor_type = Tensor<T, Shape<dims...>>;
   return fill<tensor_type>(value);
}
template <size_type... dims>
[[nodiscard]] auto fill(float value) -> Tensor<float, Shape<dims...>> {
   return fill<float, dims...>(value);
}

template <class T, size_type... dims>
[[nodiscard]] auto zeros() -> Tensor<T, Shape<dims...>> {
   using tensor_type = Tensor<T, Shape<dims...>>;
   return zeros<tensor_type>();
}
template <size_type... dims>
[[nodiscard]] auto zeros() -> Tensor<float, Shape<dims...>> {
   return zeros<float, dims...>();
}

template <class T, size_type... dims>
[[nodiscard]] auto ones() -> Tensor<T, Shape<dims...>> {
   using tensor_type = Tensor<T, Shape<dims...>>;
   return ones<tensor_type>();
}
template <size_type... dims>
[[nodiscard]] auto ones() -> Tensor<float, Shape<dims...>> {
   return ones<float, dims...>();
}

template <class T, size_type... dims>
[[nodiscard]] auto iota(T value = T(0)) -> Tensor<T, Shape<dims...>> {
   using tensor_type = Tensor<T, Shape<dims...>>;
   return iota<tensor_type>(value);
}
template <size_type... dims>
[[nodiscard]] auto iota(float value = 0.) -> Tensor<float, Shape<dims...>> {
   return iota<float, dims...>(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Functions

/// \fn min
/// Returns the maximum of an expression
template <class E>
   requires isExpr<std::remove_cvref_t<E>>
auto min(E &&expr) {
   using expr_type = std::remove_cvref_t<E>;
   return UnaryExpr<typename expr_type::node_type, functional::Min>(
       expr.node());
}

/// \fn max
/// Return the maximum of an tensor or an expression
template <class E>
   requires isExpr<std::remove_cvref_t<E>>
auto max(E &&expr) {
   using expr_type = std::remove_cvref_t<E>;
   return UnaryExpr<typename expr_type::node_type, functional::Max>(
       expr.node());
}

/// \fn abs
/// Returns the absolute value of a scalar, a tensor or an expression
template <class E>
   requires isExpr<std::remove_cvref_t<E>>
auto abs(E &&expr) {
   using expr_type = std::remove_cvref_t<E>;
   return UnaryExpr<typename expr_type::node_type, functional::Abs>(
       expr.node());
}

/// \fn sqrt
/// Returns the square root of a scalar, a tensor or an expression
template <class E>
   requires isExpr<std::remove_cvref_t<E>>
auto sqrt(E &&expr) {
   using expr_type = std::remove_cvref_t<E>;
   return UnaryExpr<typename expr_type::node_type, functional::Sqrt>(
       expr.node());
}

} // namespace ten

#endif
