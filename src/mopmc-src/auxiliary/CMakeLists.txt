set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Sorting.cpp"
        "${dir}/Sorting.h"
        "${dir}/Lincom.cpp"
        "${dir}/Lincom.h"
)

set(Sources "${Sources}" PARENT_SCOPE)