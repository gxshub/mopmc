//
// Created by guoxin on 8/11/23.
//

#ifndef MOPMC_CUDAFUNCTIONS_H
#define MOPMC_CUDAFUNCTIONS_H

namespace mopmc {
    namespace functions {
        namespace cuda {

            int aggregateLauncher(const double *w, const double *x, double *y, int numRows, int numObjs);

            int maxValueLauncher1(double *y, double *x, int *enabledActions, int *pi, int arrCount, int numRows);

            __attribute__((unused)) int maxValueLauncher2(double *y, double *x, int *enabledActions, int *pi, int *bpi, int arrCount);

            int binaryMaskingLauncher(const int *csrOffsets, const int *rowGroupIndices, const int *row2RowGroupIndices,
                                      const int *pi, int *masking4rows, int *masking4nnz, int arrCount);
            int row2RowGroupLauncher(const int *row2RowGroupMapping, int *x, int arrCount);

            template<typename T>
            struct is_not_zero {
                __host__ __device__ bool operator()(const T x) {
                    return (x != 0);
                }
            };

            __attribute__((unused)) int absLauncher(const double *x, int k);

        }// namespace cuda
    }// namespace functions
};   // namespace mopmc

#endif//MOPMC_CUDAFUNCTIONS_H
