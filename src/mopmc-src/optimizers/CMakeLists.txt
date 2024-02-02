set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/FrankWolfeInnerOptimizer.cpp"
        "${dir}/FrankWolfeInnerOptimizer.h"
        "${dir}/FrankWolfeOuterOptimizer.cpp"
        "${dir}/FrankWolfeOuterOptimizer.h"
        "${dir}/SeparationHyperplaneOptimizer.cpp"
        "${dir}/SeparationHyperplaneOptimizer.h"
)

set(Sources "${Sources}" PARENT_SCOPE)