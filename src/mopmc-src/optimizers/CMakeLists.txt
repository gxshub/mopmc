set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/FrankWolfeInnerPolytope.cpp"
        "${dir}/FrankWolfeInnerPolytope.h"
        "${dir}/FrankWolfeOuterPolytope.cpp"
        "${dir}/FrankWolfeOuterPolytope.h"
        "${dir}/LinOpt.cpp"
        "${dir}/LinOpt.h"
        "${dir}/LineSearch.cpp"
        "${dir}/LineSearch.h"
)

set(Sources "${Sources}" PARENT_SCOPE)