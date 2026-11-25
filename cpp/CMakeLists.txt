file(GLOB MOCAT_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(mocat_core ${MOCAT_SOURCES})
target_include_directories(mocat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mocat_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

add_executable(mocat6rd apps/main.cpp)
target_link_libraries(mocat6rd PRIVATE mocat_core)
set_target_properties(mocat6rd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

if(MOCAT_BUILD_TESTS)
  add_executable(mocat_tests tests/test_main.cpp)
  target_link_libraries(mocat_tests PRIVATE mocat_core)
  set_target_properties(mocat_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  add_test(NAME mocat_tests COMMAND mocat_tests)
endif()
