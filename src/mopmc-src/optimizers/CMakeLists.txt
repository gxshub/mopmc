set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/FrankWolfeMethod.cpp"
        "${dir}/FrankWolfeMethod.h"
        "${dir}/HalfspacesIntersection.cpp"
        "${dir}/HalfspacesIntersection.h"
        "${dir}/MinimumNormPoint.cpp"
        "${dir}/MinimumNormPoint.h"
        "${dir}/MaximumMarginSeparationHyperplane.cpp"
        "${dir}/MaximumMarginSeparationHyperplane.h"
        "${dir}/ProjectedGradient.cpp"
        "${dir}/ProjectedGradient.h"
        #"${dir}/ProjectedGradientV1.cpp"
        #"${dir}/ProjectedGradientV1.h"
)

set(Sources "${Sources}" PARENT_SCOPE)