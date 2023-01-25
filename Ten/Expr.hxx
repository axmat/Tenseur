#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <Ten/Types.hxx>

namespace ten {

// FIXME This may be useful for getting Functional::Apply to work?
// Keep it here for now. F is the function type or Inner Function type
// Expression must accept a FunctionWrapper<...> or
// FunctionWrapper<..>::OuterFunctionWrapper<...>
template <template <class...> class F, class... Args> struct FunctionWrapper {
   template <template <template <class...> class /*F*/, class... /*Args*/>
             class G>
   struct OuterFunctionWrapper {};
};

namespace details {
// Node shape
template <class> struct NodeWrapper;

template <class T> struct NodeWrapper<ScalarNode<T>> {
   static auto ptr(const std::shared_ptr<ScalarNode<T>> &node) {
      return node.get();
   }
};

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct NodeWrapper<TensorNode<T, Shape, Order, Storage, Allocator>> {
   static auto
   shape(const std::shared_ptr<TensorNode<T, Shape, Order, Storage, Allocator>>
             &node) {
      return node.get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<TensorNode<T, Shape, Order, Storage, Allocator>>
           &node) {
      return node.get();
   }
};

template <class L, class R, class O, template <typename...> class F,
          typename... Args>
struct NodeWrapper<BinaryNode<L, R, O, F, Args...>> {
   static auto
   shape(const std::shared_ptr<BinaryNode<L, R, O, F, Args...>> &node) {
      return node.get()->node().get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<BinaryNode<L, R, O, F, Args...>> &node) {
      return node.get()->node().get();
   }
};
} // namespace details

namespace details {
// Node type
template <class> struct OutputNodeType;

template <class T> struct OutputNodeType<ScalarNode<T>> {
   using type = ScalarNode<T>;
};

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct OutputNodeType<TensorNode<T, Shape, Order, Storage, Allocator>> {
   using type = TensorNode<T, Shape, Order, Storage, Allocator>;
};

template <class L, class R, class Output, template <typename...> class F,
          class... Args>
struct OutputNodeType<::ten::BinaryNode<L, R, Output, F, Args...>> {
   using type =
       typename ::ten::BinaryNode<L, R, Output, F, Args...>::output_node_type;
};
} // namespace details

// TODO Remove this, Value type of an expression
template <class E, template <typename...> class Func, class... Args>
struct InputNode;

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator, template <typename...> class Func, class... Args>
struct InputNode<Tensor<T, Shape, Order, Storage, Allocator>, Func, Args...> {
   using type = TensorNode<T, Shape, Order, Storage, Allocator>;
};

template <class T, template <typename...> class Func, class... Args>
struct InputNode<Scalar<T>, Func, Args...> {
   using type = ScalarNode<T>;
};

template <class L, class R, template <typename...> class G, class... GArgs,
          template <typename...> class Func, class... Args>
struct InputNode<::ten::BinaryExpr<L, R, G, GArgs...>, Func, Args...> {
   using type = typename ::ten::BinaryExpr<L, R, G, GArgs...>::output_node_type;
};

template <class L, class R, class Output, template <typename...> class G,
          class... GArgs, template <typename...> class Func, class... Args>
struct InputNode<::ten::BinaryNode<L, R, Output, G, GArgs...>, Func, Args...> {
   using type =
       typename ::ten::BinaryNode<L, R, Output, G, GArgs...>::output_node_type;
};

// \class UnaryNode
// Apply a function to a ten::Tensor or a ten::Scalar
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class UnaryNode {
 public:
   using func_type = Func<Input, Output, Args...>;

 private:
   /// Optional function type
   std::optional<func_type> _func;
   //// Output value
   std::shared_ptr<Output> _value = nullptr;

 public:
   /// Construct a unary node if the function doesn't take additional parameters
   UnaryNode() noexcept
      requires(!func_type::isParametric())
   {}

   /// Construct a unary node if the function take additional parameters.
   /// The parameters of the functions args of type FuncArgs are forwarded
   /// to the constructor of the function when necessary.
   template <typename... FuncArgs>
   UnaryNode(FuncArgs... args) noexcept
      requires(func_type::isParametric())
       : _func(func_type(std::forward<FuncArgs>(args)...)) {}

   /// Returns whether the functions has parameters
   constexpr inline bool isParametric() { return func_type::isParametric(); }

   /// Returns whether a function object has been initialized and stored
   /// in the node.
   inline bool hasFunc() const { return _func.has_value(); }

   const func_type &func() { return _func.value(); }

   [[nodiscard]] inline std::shared_ptr<Output> value() { return _value; }

   /// Evaluated the expression for a parametrized function
   template <class T>
   void callFunc(T &&input) noexcept
      requires(func_type::isParametric())
   {
      if constexpr (Output::isStatic()) {
         _value = std::make_shared<Output>();
      } else {
         _value = std::make_shared<Output>(input.shape());
      }
      _func.value().call(std::forward<T>(input), *_value.get());
   }

   /// Evaluate the expression for a non parametric function
   /// T is a ten::TensorNode or ten::ScalarNode
   /// args are the parameters of the function.
   template <class T>
   void callFunc(T &&input) noexcept
      requires(!func_type::isParametric())
   {
      // scalar
      if constexpr (isScalarNode<Output>::value) {
         _value = std::make_shared<Output>();
      } else { // tensor
         if constexpr (Output::isStatic()) {
            _value = std::make_shared<Output>();
         } else {
            _value = std::make_shared<Output>(input.shape());
         }
      }
      func_type::call(std::forward<T>(input), *_value.get());
   }
};

/// \class UnaryExpr
/// Unary expression.
template <typename E, template <class...> class Func, typename... Args>
class UnaryExpr : ::ten::Expr<UnaryExpr<E, Func, Args...>> {
 public:
   /// \typedef input_type
   /// Type of the input type of the function
   using input_type = typename InputNode<E, Func, Args...>::type;
   // input_type is ScalarNode or TensorNode
   static_assert(isScalarNode<input_type>::value ||
                     isTensorNode<input_type>::value,
                 "Input type of the function of a UnaryExpr must be a "
                 "ScalarNode or TensorNode.");

   /// \typedef output_type
   /// Type of the output type of the function
   using output_type = typename Func<input_type>::output_type;
   // output_type is ScalarNode or TensorNode
   static_assert(isScalarNode<output_type>::value ||
                     isTensorNode<output_type>::value,
                 "Output type of the function of a UnaryExpr must be a "
                 "ScalarNode or TensorNode.");

   /// \typedef node_type
   /// Type of the node of the unary expresion
   using node_type = UnaryNode<input_type, output_type, Func, Args...>;

   /// \typedef parent_type
   /// Parent node type
   using parent_type = typename E::node_type;

   /// \typedef expr_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<isTensorNode<output_type>::value,
                                             typename output_type::tensor_type,
                                             typename output_type::scalar_type>;

 private:
   bool _evaluated = false;
   /// Shared ptr to the node
   std::shared_ptr<node_type> _node;
   /// Shared ptr to the parent node
   std::shared_ptr<parent_type> _parent;

 public:
   /// Construct a ten::UnaryNode from an expression
   explicit UnaryExpr(const E &e) noexcept
       : _node(std::make_shared<node_type>()), _parent(e.node()) {
      // std::cout << "UnaryExpr(const E&)" << std::endl;
   }

   /* TODO explicit UnaryExpr(E&& e) noexcept
    : _node(std::make_shared<node_type>()), _parent() {
    std::cout << "UnaryExpr(E&&)" << std::endl;
   }*/

   // Returns the shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns the parent node (input node)
   [[nodiscard]] std::shared_ptr<parent_type> parent() const { return _parent; }

   /// Returns whether the expression is evaluated
   bool evaluated() const { return _evaluated; }

   /// Returns the shared pointer to the node of the evaluated expression
   [[nodiscard]] std::shared_ptr<output_type> valueNode() {
      return _node.get()->value();
   }

   /// Returns the scalar or tensor value of the evaluated expression
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_node.get()->value());
   }

   /// Evaluate a unary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression.
   [[maybe_unused]] auto eval() -> evaluated_type {
      if (_evaluated)
         return value();

      if constexpr (isTensor<E>::value || isScalar<E>::value) {
         _node.get()->callFunc(*_parent.get());
      } else if constexpr (isBinaryExpr<E>::value) {
         // FIXME move these to the node like in BinaryNode and BinaryExpr
         _parent.get()->eval();
         _node.get()->callFunc(*_parent.get()->valueNode());
      }

      _evaluated = true;
      return value();
   }
};

// \class BinaryNode
// Node of a binary expresion
// Left and Right can be ScalarNode, TensorNode or BinaryNode
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
class BinaryNode {
 public:
   /// Left input type
   using left_node_type = details::OutputNodeType<Left>::type;

   /// Right input type
   using right_node_type = details::OutputNodeType<Right>::type;

   /// Output type
   using output_node_type = Output;

   using func_type = Func<left_node_type, right_node_type, Output, Args...>;
   // using shape_type = typename Output::shape_type;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<isScalarNode<Output>::value,
                                             typename Output::scalar_type,
                                             typename Output::tensor_type>;

 private:
   std::optional<func_type> _func = std::nullopt;
   // std::optional<typename Output::shape_type> _shape = std::nullopt;
   std::shared_ptr<Output> _value = nullptr;
   std::shared_ptr<Left> _left;
   std::shared_ptr<Right> _right;

 public:
   BinaryNode() {}

   BinaryNode(const std::shared_ptr<Left> &left,
              const std::shared_ptr<Right> &right) noexcept
      requires(!func_type::isParametric())
       : _left(left), _right(right) {}

   template <typename... FuncArgs>
   BinaryNode(const std::shared_ptr<Left> &left,
              const std::shared_ptr<Right> &right, FuncArgs... args) noexcept
      requires(func_type::isParametric())
       : _func(func_type(std::forward<FuncArgs>(args)...)), _left(left),
         _right(right) {}

   constexpr inline bool isParametric() { return func_type::isParametric(); }

   inline bool hasFunc() const { return _func.has_value(); }

   const func_type &func() { return _func.value(); }

   [[nodiscard]] inline std::shared_ptr<Output> node() { return _value; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _value; }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_value);
   }

   auto eval() noexcept -> evaluated_type {
      if (_value)
         return evaluated_type(_value);

      // Evaluate LeftExpr
      if constexpr (::ten::isBinaryNode<Left>::value) {
         _left.get()->eval();
      }
      // Evaluate RightExpr
      if constexpr (::ten::isBinaryNode<Right>::value) {
         _right.get()->eval();
      }

      if constexpr (Output::isStatic()) {
         _value.reset(new Output());
      } else {
         if constexpr (!::ten::isScalarNode<Left>::value &&
                       !::ten::isScalarNode<Right>::value) {
            _value.reset(new Output(func_type::outputShape(
                ::ten::details::NodeWrapper<Left>::shape(_left),
                ::ten::details::NodeWrapper<Right>::shape(_right))));
         } else {
            if constexpr (::ten::isScalarNode<Left>::value &&
                          !::ten::isScalarNode<Right>::value) {
               _value.reset(new Output(func_type::outputShape(
                   ::ten::details::NodeWrapper<Right>::shape(_right))));
            }
            if constexpr (!::ten::isScalarNode<Left>::value &&
                          ::ten::isScalarNode<Right>::value) {
               _value.reset(new Output(func_type::outputShape(
                   ::ten::details::NodeWrapper<Left>::shape(_left))));
            }
         }
      }

      // Call F
      func_type::call(*details::NodeWrapper<Left>::ptr(_left),
                      *details::NodeWrapper<Right>::ptr(_right), *_value.get());

      return evaluated_type(_value);
   }
};

/// \class BinaryExpr
/// Binary expression
// Left and Right can be ScalarNode, TensorNode or BinaryExpr
// left is std::shared_ptr<Left> and right is std::shared_ptr<Right>
template <typename Left, typename Right, template <typename...> class Func,
          typename... Args>
class BinaryExpr : ::ten::Expr<BinaryExpr<Left, Right, Func, Args...>> {
 public:
   /// Left input type
   using left_node_type = details::OutputNodeType<Left>::type;

   /// Right input type
   using right_node_type = details::OutputNodeType<Right>::type;

   /// output_node_type is ScalarNode or TensorNode
   using output_node_type =
       typename Func<left_node_type, right_node_type>::output_type;

   // Node type
   using node_type = BinaryNode<Left, Right, output_node_type, Func, Args...>;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<isScalarNode<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

 private:
   std::shared_ptr<node_type> _node;

 public:
   /// Construct a BinaryExpr from an expression
   explicit BinaryExpr(const std::shared_ptr<Left> &left,
                       const std::shared_ptr<Right> &right) noexcept
       : _node(std::make_shared<node_type>(left, right)) {}

   //, _left_input(left.node()), _right_input(right.node()) {}
   /// Returns a shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns a shared pointer to the left input
   /*[[nodiscard]] std::shared_ptr<left_input_type> leftInput() const {
      return _left_input;
   }*/

   /// Returns a shared pointer to the right input
   /*[[nodiscard]] std::shared_ptr<right_input_type> rightInput() const {
      return _right_input;
   }*/

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _node.get()->evaluated(); }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_node.get()->node());
   }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() -> evaluated_type {
      return _node.get()->eval();
   }
};

} // namespace ten

#endif
