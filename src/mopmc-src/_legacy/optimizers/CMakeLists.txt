set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/FrankWolfe.cpp"
        "${dir}/FrankWolfe.h"
        "${dir}/LinOpt.cpp"
        "${dir}/LinOpt.h"
        "${dir}/ProjectedGradientDescent.cpp"
        "${dir}/ProjectedGradientDescent.h"
        "${dir}/ProjectedGradientV1.cpp"
        "${dir}/ProjectedGradientV1.h"
)

set(Sources "${Sources}" PARENT_SCOPE)