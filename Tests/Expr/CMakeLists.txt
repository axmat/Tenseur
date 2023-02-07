add_executable(TestExpr TestExpr.cxx)
target_link_libraries(TestExpr gtest_main)

target_include_directories(TestExpr PRIVATE ${PROJECT_SOURCE_DIR})
gtest_discover_tests(TestExpr)

