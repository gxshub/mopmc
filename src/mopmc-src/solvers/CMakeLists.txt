set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseValueIteration.h"
        "${dir}/CudaFunctions.cu"
        "${dir}/CudaFunctions.h"
        "${dir}/CudaValueIteration.cu"
        "${dir}/CudaValueIteration.cuh"
        "${dir}/CudaWarmUp.cu"
        "${dir}/CudaWarmUp.h"
        "${dir}/ValueIteration.cpp"
        "${dir}/ValueIteration.h"
)

set(Sources "${Sources}" PARENT_SCOPE)