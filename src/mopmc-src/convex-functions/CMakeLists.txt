set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseConvexFunction.cpp"
        "${dir}/BaseConvexFunction.h"
        "${dir}/MSE.cpp"
        "${dir}/MSE.h"
        "${dir}/Variance.cpp"
        "${dir}/Variance.h"
)

set(Sources "${Sources}" PARENT_SCOPE)