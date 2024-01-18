set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ConvexQuery.cpp"
        "${dir}/ConvexQuery.h"
)

set(Sources "${Sources}" PARENT_SCOPE)