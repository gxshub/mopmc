set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/FrankWolfe.cpp"
        "${dir}/FrankWolfe.h"
        "${dir}/ProjectedGradientDescent.cpp"
        "${dir}/ProjectedGradientDescent.h"
)

set(Sources "${Sources}" PARENT_SCOPE)