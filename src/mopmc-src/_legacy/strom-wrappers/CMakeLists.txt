# Sub dir needs to be at the top of the parent file
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/StormModelCheckingWrapper.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
