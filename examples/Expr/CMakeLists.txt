add_executable(TestExpr TestExpr.cxx)
target_link_libraries(TestExpr ${BLAS_LIBRARIES})
target_include_directories(TestExpr PRIVATE ${PROJECT_SOURCE_DIR})
