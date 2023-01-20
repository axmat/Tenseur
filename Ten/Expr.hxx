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

// Value type of an expression
template <class E, template <typename...> class Func, class... Args>
struct InputNode;

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator, template <typename...> class Func, class... Args>
struct InputNode<Tensor<T, Shape, Order, Storage, Allocator>, Func, Args...> {
   using type = TensorNode<T, Shape, Order, Storage, Allocator>;
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
      std::cout << "UnaryExpr(const E&)" << std::endl;
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
      }

      _evaluated = true;
      return value();
   }
};

// \class BinaryNode
// Node of a binary expresion
template <class LeftInput, class RightInput, class Output,
          template <typename...> class Func, typename... Args>
class BinaryNode {
 public:
   using func_type = Func<LeftInput, RightInput, Output, Args...>;
   using shape_type = typename Output::shape_type;

 private:
   std::optional<func_type> _func = std::nullopt;
   std::optional<typename Output::shape_type> _shape = std::nullopt;
   std::shared_ptr<Output> _value = nullptr;

 public:
   BinaryNode() noexcept
      requires(!func_type::isParametric())
   {}

   template <typename... FuncArgs>
   BinaryNode(FuncArgs... args) noexcept
      requires(func_type::isParametric())
       : _func(func_type(std::forward<FuncArgs>(args)...)) {}

   constexpr inline bool isParametric() { return func_type::isParametric(); }

   inline bool hasFunc() const { return _func.has_value(); }

   const func_type &func() { return _func.value(); }

   [[nodiscard]] inline std::shared_ptr<Output> value() { return _value; }

   void callFunc(const LeftInput &leftInput,
                 const RightInput &rightInput) noexcept
      requires(!func_type::isParametric())
   {
      if constexpr (Output::isStatic()) {
         _value.reset(new Output());
      } else {
         _value.reset(new Output(
             func_type::outputShape(leftInput.shape(), rightInput.shape())));
      }
      func_type::call(leftInput, rightInput, *_value.get());
   }
};

/// \class BinaryExpr
/// Binary expression
template <typename LeftExpr, typename RightExpr,
          template <typename...> class Func, typename... Args>
class BinaryExpr : ::ten::Expr<BinaryExpr<LeftExpr, RightExpr, Func, Args...>> {
 public:
   /// Left input type
   using left_node_type = typename InputNode<LeftExpr, Func, Args...>::type;
   /// Right input type
   using right_node_type = typename InputNode<RightExpr, Func, Args...>::type;

   /// Output type
   using output_node_type =
       typename Func<left_node_type, right_node_type>::output_type;
   /// output_node_type is ScalarNode or TensorNode
   static_assert(
       isScalarNode<output_node_type>::value ||
           isTensorNode<output_node_type>::value,
       "Output node type of BinaryExpr must be a ScalarNode or TensorNode.");

   /// \typedef node_type
   /// Node type
   using node_type = BinaryNode<left_node_type, right_node_type,
                                output_node_type, Func, Args...>;

   /// \typedef left_input_type
   /// Left input node type
   using left_input_type = typename LeftExpr::node_type;

   /// \typedef right_input_type
   /// Right input node type
   using right_input_type = typename RightExpr::node_type;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<isScalarNode<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

 private:
   bool _evaluated = false;
   std::shared_ptr<node_type> _node;
   std::shared_ptr<left_input_type> _left_input;
   std::shared_ptr<right_input_type> _right_input;

 public:
   /// Construct a BinaryExpr from an expression
   explicit BinaryExpr(const LeftExpr &left, const RightExpr &right) noexcept
       : _node(std::make_shared<node_type>()), _left_input(left.node()),
         _right_input(right.node()) {}

   /// Returns a shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns a shared pointer to the left input
   [[nodiscard]] std::shared_ptr<left_input_type> leftInput() const {
      return _left_input;
   }

   /// Returns a shared pointer to the right input
   [[nodiscard]] std::shared_ptr<right_input_type> rightInput() const {
      return _right_input;
   }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _evaluated; }

   /// Returns a shared pointer to the node of the evaluated expression
   [[nodiscard]] std::shared_ptr<output_node_type> valueNode() {
      return _node.get()->value();
   }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(valueNode());
   }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() -> evaluated_type {
      if (_evaluated)
         return value();

      // TODO Evaluate LeftExpr
      // TODO Evaluate RightExpr
      static_assert(
          isTensor<LeftExpr>::value || isScalar<LeftExpr>::value ||
              isTensor<RightExpr>::value || isScalar<RightExpr>::value,
          "LeftExpr and RightExpr must be evaluated before evaluating "
          "BinaryExpr.");
      // Evaluate Expr
      _node.get()->callFunc(*_left_input.get(), *_right_input.get());

      _evaluated = true;
      return value();
   }
};

} // namespace ten

#endif
