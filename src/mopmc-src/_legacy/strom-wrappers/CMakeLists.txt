# Sub dir needs to be at the top of the parent file
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/StormModelCheckingWrapper.h"
        "${dir}/StormModelBuildingWrapper.cpp"
        "${dir}/StormModelBuildingWrapper.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
