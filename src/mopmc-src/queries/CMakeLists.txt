set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.cpp"
        "${dir}/AchievabilityQuery.h"
        "${dir}/BaseQuery.h"
        "${dir}/ConvexQuery.cpp"
        "${dir}/ConvexQuery.h"
        "${dir}/ConvexQuery2.cpp"
        "${dir}/ConvexQuery2.h"
)

set(Sources "${Sources}" PARENT_SCOPE)