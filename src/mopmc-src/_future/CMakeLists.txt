set(dir "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(hybrid-computing)
add_subdirectory(queries)

list(
        APPEND Sources
)

set(Sources "${Sources}" PARENT_SCOPE)
