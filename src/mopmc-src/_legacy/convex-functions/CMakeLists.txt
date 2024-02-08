set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/EuclideanDistance.cpp"
        "${dir}/EuclideanDistance.h"
        "${dir}/KLDistance.cpp"
        "${dir}/KLDistance.h"
        "${dir}/SignedKLEuclidean.cpp"
        "${dir}/SignedKLEuclidean.h"
        "${dir}/TotalReLU.cpp"
        "${dir}/TotalReLU.h"
)

set(Sources "${Sources}" PARENT_SCOPE)