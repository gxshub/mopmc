set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(auxiliary)
add_subdirectory(convex-functions)
add_subdirectory(optimizers)
add_subdirectory(queries)
add_subdirectory(solvers)
add_subdirectory(storm-wrappers)
add_subdirectory(_future)
add_subdirectory(_legacy)

list(
        APPEND Sources
        "${dir}/QueryData.h"
        "${dir}/QueryOptions.h"
        "${dir}/Runner.cpp"
        "${dir}/Runner.h"
        "${dir}/Transformation.cpp"
        "${dir}/Transformation.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
