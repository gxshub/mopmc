set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/UnconstrainedConvexQuery.cpp"
        "${dir}/UnconstrainedConvexQuery.h"
)

set(Sources "${Sources}" PARENT_SCOPE)