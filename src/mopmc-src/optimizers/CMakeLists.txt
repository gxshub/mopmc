set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/FrankWolfeInnerOptimizer.cpp"
        "${dir}/FrankWolfeInnerOptimizer.h"
        "${dir}/HalfspacesIntersectionCheck.cpp"
        "${dir}/HalfspacesIntersectionCheck.h"
        "${dir}/ProjectedGradient.cpp"
        "${dir}/ProjectedGradient.h"
        "${dir}/SeparationHyperplaneOptimizer.cpp"
        "${dir}/SeparationHyperplaneOptimizer.h"
)

set(Sources "${Sources}" PARENT_SCOPE)