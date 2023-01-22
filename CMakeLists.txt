cmake_minimum_required(VERSION 3.2)

project(Tenseur VERSION 1.0.0
                DESCRIPTION "A header only C++20 tensor library."
                HOMEPAGE_URL "https://github.com/axmat/Tenseur")

include(GNUInstallDirs)

# Compiler options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 -march=native")

# Library
add_library(Tenseur INTERFACE)
target_include_directories(
  Tenseur
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Ten>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# TODO Debug
#message("Project source dir = ${PROJECT_SOURCE_DIR}")

# Compiler features
target_compile_features(Tenseur INTERFACE cxx_std_20)

# Target paths
install(TARGETS Tenseur
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Version
include(CMakePackageConfigHelpers)
write_basic_package_version_file("TenseurConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)


configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Targets to install
install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Files to install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake)

# Install directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Ten DESTINATION include)

# Build the examples
option(TENSEUR_EXAMPLES "Build examples." OFF)
if (TENSEUR_EXAMPLES)
  add_subdirectory(examples)
endif()

