cmake_minimum_required(VERSION 3.15...3.27)
project(pgeof)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(pgeof_ext NOMINSIZE STABLE_ABI LTO src/pgeof_ext.cpp)

nanobind_add_stub(
   pgeof_ext_stub
   MODULE pgeof_ext
   OUTPUT pgeof_ext.pyi
   MARKER_FILE py.typed
   PYTHON_PATH $<TARGET_FILE_DIR:pgeof_ext>
   DEPENDS pgeof_ext
)

# All lib are headeer only. 
# it's faster to include like this than using exported targets 
# (i.e add_subdirectories(...))
target_include_directories(pgeof_ext PRIVATE "include" "third_party/eigen" "third_party/nanoflann/include" "third_party/taskflow") 

install(TARGETS pgeof_ext LIBRARY DESTINATION pgeof)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pgeof_ext.pyi ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION pgeof)