//
// Created by guoxin on 15/11/23.
//

#ifndef MOPMC_CUDAVALUEITERATION_CUH
#define MOPMC_CUDAVALUEITERATION_CUH


#include "BaseValueIteration.h"
#include "mopmc-src/QueryData.h"
#include <Eigen/Sparse>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <storm/storage/SparseMatrix.h>

namespace mopmc {
    namespace value_iteration {
        namespace gpu {
            template<typename ValueType>
            class CudaValueIterationHandler : public mopmc::value_iteration::BaseVIHandler<ValueType> {
            public:
                explicit CudaValueIterationHandler(mopmc::QueryData<ValueType, int> *queryData);

                int initialize() override;
                int exit() override;

                int valueIteration(const std::vector<double> &w) override;
                int valueIterationPhaseOne(const std::vector<double> &w, bool toHost = false);
                int valueIterationPhaseTwo();
                //__attribute__((unused)) int valueIterationPhaseTwo_deprecated();

                const std::vector<double> &getResults() const override {
                    return results;
                }

            private:
                mopmc::QueryData<ValueType, int> *data;
                Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix;
                std::vector<ValueType> flattenRewardVector;
                std::vector<int> scheduler;
                std::vector<int> rowGroupIndices;
                std::vector<int> row2RowGroupMapping;
                std::vector<double> weightedValueVector;
                std::vector<double> results;
                int iniRow{};
                int nobjs{};

                int *dA_csrOffsets{}, *dA_columns{}, *dA_rows_backup{};
                int *dB_csrOffsets{}, *dB_columns{}, *dB_rows_backup{};
                int *dRowGroupIndices{}, *dRow2RowGroupMapping{}, *dScheduler{};
                int *dMasking_nnz{}, *dMasking_nrows{}, *dMasking_tiled{};
                double *dA_values{}, *dB_values{};
                double *dR{}, *dRi{}, *dRPart{};
                double *dW{}, *dRw{};
                double *dX{}, *dX1{}, *dY{}, *dZ{}, *dZ1{};
                double *dResult{};
                int A_nnz{}, A_ncols{}, A_nrows{};
                int B_nnz{}, B_ncols{}, B_nrows{};
                int Z_ncols{}, Z_nrows{}, Z_ld{};

                double alpha = 1.0, alpha2 = -1.0, beta = 1.0;
                double eps = 1.0, maxEps{}, tolerance = 1.0e-8;
                int maxIter = 2000, maxInd = 0;
                int iteration{};

                //CUSPARSE APIs
                cublasHandle_t cublasHandle = nullptr;
                cusparseHandle_t handle = nullptr;
                cusparseSpMatDescr_t matA{}, matB{};
                cusparseDnMatDescr_t matZ{}, matZ1{};
                cusparseDnVecDescr_t vecRw{}, vecX{}, vecX1{}, vecY{};
                void *dBuffer = nullptr, *dBufferB = nullptr;
                size_t bufferSize = 0, bufferSizeB = 0;
            };
        }// namespace gpu
    }// namespace value_iteration
}// namespace mopmc

#endif//MOPMC_CUDAVALUEITERATION_CUH
