set(dir "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(convex-functions)
add_subdirectory(model-checking)
add_subdirectory(optimizers)
add_subdirectory(queries)
add_subdirectory(solvers)

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
