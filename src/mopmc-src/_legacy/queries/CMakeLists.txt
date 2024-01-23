set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ConvexQueryV0.cpp"
        "${dir}/ConvexQueryV0.h"
)

set(Sources "${Sources}" PARENT_SCOPE)