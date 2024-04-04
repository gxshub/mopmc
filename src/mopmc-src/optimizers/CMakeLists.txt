set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/FrankWolfeInnerOptimizer.cpp"
        "${dir}/FrankWolfeInnerOptimizer.h"
        "${dir}/HalfspacesIntersection.cpp"
        "${dir}/HalfspacesIntersection.h"
        "${dir}/MinimumNormPoint.cpp"
        "${dir}/MinimumNormPoint.h"
        "${dir}/MaximumMarginSeparationHyperplane.cpp"
        "${dir}/MaximumMarginSeparationHyperplane.h"
        "${dir}/ProjectedGradient.cpp"
        "${dir}/ProjectedGradient.h"
)

set(Sources "${Sources}" PARENT_SCOPE)