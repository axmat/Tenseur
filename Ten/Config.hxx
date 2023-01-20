#ifndef TENSEUR_CONFIG_HXX
#define TENSEUR_CONFIG_HXX

#include <optional>
#include <ostream>

namespace ten {

// Architecture
enum class Arch {
  darwin,
  x64,
  unknown
};

template <Arch arch> struct is_supported_arch {
  static constexpr bool value = arch != Arch::unknown;
};

constexpr Arch get_arch_type() {
#if defined(__APPLE__)
  return Arch::darwin;
#endif
#if defined(__linux__)
  return Arch::x64;
#endif
  return Arch::unknown;
}
static constexpr Arch arch = get_arch_type();

static_assert(is_supported_arch<arch>::value, "Unsuported architecture.");

} // namespace ten

#endif
