//
// Created by guoxin on 15/11/23.
//

#include "CudaValueIteration.cuh"
#include "CuFunctions.h"
#include "../solvers/WarmUp.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

namespace mopmc {
    namespace value_iteration {
        namespace gpu {

            template<typename ValueType>
            CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(QueryData<ValueType, int> *queryData) :
                    data(queryData),
                    transitionMatrix_(queryData->transitionMatrix),
                    rowGroupIndices_(queryData->rowGroupIndices),
                    row2RowGroupMapping_(queryData->row2RowGroupMapping),
                    flattenRewardVector_(queryData->flattenRewardVector),
                    scheduler_(queryData->scheduler),
                    iniRow_(queryData->initialRow),
                    nobjs(queryData->objectiveCount)
            {
                A_nnz = transitionMatrix_.nonZeros();
                A_ncols = transitionMatrix_.cols();
                A_nrows = transitionMatrix_.rows();
                B_ncols = A_ncols;
                B_nrows = B_ncols;
                C_nrows = B_ncols;
                C_ncols = 1;// nobjs; //todo just need endObj - beginObj rows
                C_ld = C_nrows;
                results_.resize(nobjs+1);
                //Assertions
                assert(A_ncols == scheduler_.size());
                assert(flattenRewardVector_.size() == A_nrows * nobjs);
                assert(rowGroupIndices_.size() == A_ncols + 1);
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::initialize() {
                //GPU warm up
                mopmc::kernels::launchWarmupKernel();
                std::cout << ("____ CUDA INITIALIZING ____\n");
                // cudaMalloc CONSTANTS -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (A_nrows + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dA_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRowGroupIndices, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRow2RowGroupMapping, A_nrows * sizeof(int)))
                // cudaMalloc Variables -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dResult, (nobjs + 1) * sizeof(double)))
                // cudaMalloc PHASE B-------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dB_csrOffsets, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dB_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nrows, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nnz, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRi, B_nrows * sizeof(double))) // TR TODO: extra allocated mem which we possibly won't use in the hybrid version
                CHECK_CUDA(cudaMalloc((void **) &dRj, nobjs * B_nrows * sizeof(double))) // TR TODO: ^ 
                CHECK_CUDA(cudaMalloc((void **) &dZ, C_nrows * C_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dZPrime, C_nrows * C_ncols * sizeof(double)))
                // cudaMemcpy -------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(), (A_nrows + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(), A_nnz * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(), A_nnz * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dR, flattenRewardVector_.data(), A_nrows * nobjs * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRowGroupIndices, rowGroupIndices_.data(), (A_ncols + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRow2RowGroupMapping, row2RowGroupMapping_.data(), A_nrows * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dPi, scheduler_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice));
                // NOTE. Data for dW copied in VI phase B.
                //CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
                //-------------------------------------------------------------------------
                CHECK_CUSPARSE(cusparseCreate(&handle))
                CHECK_CUBLAS(cublasCreate_v2(&cublasHandle));
                // Create sparse matrices A in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                // Crease dense matrix C
                CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C_nrows, C_ncols, C_ld, dZ, CUDA_R_64F, CUSPARSE_ORDER_COL));
                // Crease dense matrix D
                CHECK_CUSPARSE(cusparseCreateDnMat(&matD, C_nrows, C_ncols, C_ld, dZPrime, CUDA_R_64F, CUSPARSE_ORDER_COL));
                // Create dense vector X
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_ncols, dX, CUDA_R_64F))
                // Create dense vector Y
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_nrows, dY, CUDA_R_64F))
                // Create dense vector XPrime
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecXPrime, A_ncols, dXPrime, CUDA_R_64F))
                // Create dense vector Rw
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecRw, A_nrows, dRw, CUDA_R_64F))
                // allocate an external buffer if needed
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
                /*
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                        handleB, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeB))
                        */
                CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
                CHECK_CUDA(cudaMalloc(&dBufferC, bufferSizeC));

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIteration(const std::vector<double> &w) {

                this->valueIterationPhaseOne(w);
                //this->valueIterationPhaseTwo_dev();
                this->valueIterationPhaseTwo();
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w, bool toHost) {
                std::cout << "____ VI PHASE ONE ____\n" ;
                CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
                mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);

                iteration = 0;
                do {
                    // y = r
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dRw, 1, dY, 1))
                    if (iteration == 0) {
                        mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dPi, A_ncols + 1, A_nrows);
                    }
                    // y = A.x + r (r = y)
                    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                    // x' = x
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))
                    // x(s) = max_{a\in Act(s)} y(s,a), pi(s) = argmax_{a\in Act(s)} pi(s)
                    mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dPi, A_ncols + 1, A_nrows);
                    // x' = -1 * x + x'
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dXPrime, 1))
                    // max |x'|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd))
                    // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                    CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
                    ++iteration;
                    //printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iteration, maxEps);
                } while (maxEps > 1e-5 && iteration < maxIter);

                if (iteration == maxIter) {
                    std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                }
                //std::cout << "terminated after " << iteration <<" iterations.\n";
                //copy result
                thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + nobjs);
                if(toHost) {
                    CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                }

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo() {
                std::cout << "____ VI PHASE TWO (deprecated) ____\n";
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                /* @param B_nnz: number of non-zero entries in the DTMC transition matrix */
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                for (int obj = 0; obj < nobjs; obj++) {
                    thrust::copy_if(thrust::device, dR + obj * A_nrows, dR + (obj + 1) * A_nrows,
                                    dMasking_nrows, dRi, mopmc::functions::cuda::is_not_zero<double>());

                    iteration = 0;
                    do {
                        // x = ri
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX, 1));
                        // initialise x' as ri too
                        if (iteration == 0) {
                            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dXPrime, 1));
                        }
                        // x = B.x' + ri where x = ri
                        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                        // x' = -1 * x + x'
                        CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dXPrime, 1));
                        // max |x'|
                        CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd));
                        // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                        CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                        maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                        // x' = x
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dX, 1, dXPrime, 1));

                        //printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                        ++iteration;

                    } while (maxEps > 1e-5 && iteration < maxIter);
                    if (iteration == maxIter) {
                        std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                    }
                    //std::cout << "objective " << obj  << " terminated after " << iteration << " iterations\n";
                    // copy results
                    thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + obj);

                }

                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                return EXIT_SUCCESS;
            }

            // TODO this one does not work yet
            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo_dev(int beginObj, int endObj) {
                std::cout << "____ VI PHASE TWO ____\n";
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                /* @param B_nnz: number of non-zero entries in the DTMC transition matrix */
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                for (int obj = 0; obj < nobjs; obj++) {
                    thrust::copy_if(thrust::device, dR + obj * A_nrows, dR + (obj + 1) * A_nrows,
                                    dMasking_nrows, dRi, mopmc::functions::cuda::is_not_zero<double>());


                    iteration = 0;
                    maxEps = 1;
                    do {
                        // x = ri
                        //CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX, 1));
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dZ, 1));
                        // initialise x' as ri too
                        if (iteration == 0) {
                            //CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dXPrime, 1));
                            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dZPrime, 1));
                        }
                        // x = B.x' + ri where x = ri
                        /*
                        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matB, matD, &beta, matC,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeC))*/

                        CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, matD, &beta, matC, CUDA_R_64F,
                                                    CUSPARSE_SPMM_ALG_DEFAULT, &dBufferC))

                        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                        // x' = -1 * x + x'
                        CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dXPrime, 1));
                        //CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dZ, 1, dZPrime, 1));
                        // max |x'|
                        CHECK_CUBLAS(cublasIdamax(cublasHandle, B_nrows, dXPrime, 1, &maxInd));
                        //CHECK_CUBLAS(cublasIdamax(cublasHandle, C_nrows, dZPrime, 1, &maxInd));
                        // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                        CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                        //CHECK_CUDA(cudaMemcpy(&maxEps, dZPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                        maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                        // x' = x
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dX, 1, dXPrime, 1));
                        //CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dZ, 1, dZPrime, 1));

                        printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                        ++iteration;

                    } while (iteration <5);//(maxEps > 1e-5 && iteration < maxIter);
                    if (iteration == maxIter) {
                        std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                    }
                    //std::cout << "objective " << obj  << " terminated after " << iteration << " iterations\n";
                    // copy results
                    thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + obj);
                    //thrust::copy(thrust::device, dZ + iniRow_, dZ + iniRow_ + 1, dResult + obj);

                }

                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matD))
                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo_v2(int beginObj, int endObj) {
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                /* @param B_nnz: number of non-zero entries in the DTMC transition matrix */
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                //
                // TR: We should avoid this loop and group together windows of the R vector into a matrix
                // one way to do this is just by copying a portion of dR
                //
                double* dRPortion, *dMaskTiled;
                size_t bufferSizeMM;
                void *dBufferMM = nullptr;
                CHECK_CUDA(cudaMalloc((void**) &dRPortion, (endObj - beginObj) * B_nrows * sizeof(double)))
                //^^^^^^
                //GS: can replace this with a kernel function, avoiding tiling dMask
                // also tile dMasking_nrows (endObj - beginObj) times
                CHECK_CUDA(cudaMalloc((void**) &dMaskTiled, (endObj - beginObj) * A_nrows * sizeof(double)))
                for (uint i = 0; i < (endObj - beginObj); ++i) {
                    thrust::copy(thrust::device, dMasking_nrows, dMasking_nrows + A_nrows, dMaskTiled + i * A_nrows);
                }
                // create a mask of dR based on the tiled dMasking rows starting at the objective index
                thrust::copy_if(thrust::device, dR + beginObj * A_nrows, dR + endObj * A_nrows,
                                dMaskTiled, dRPortion, mopmc::functions::cuda::is_not_zero<double>());
                //vvvvvv

                // Make dRPortion into a dense matrix
                // create a new cuBlas handle => this can be preallocated todo
                double *dY2, *dX2; // TODO rename using proper convention -> insert into initialise once working
                CHECK_CUDA(cudaMalloc((void**) &dY2, (endObj - beginObj) * B_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dX2, (endObj - beginObj) * B_nrows * sizeof(double)))
                CHECK_CUDA(cudaMemset(dX2, 0, (endObj - beginObj) * B_nrows * sizeof(double)))
                cusparseDnMatDescr_t matY2, matX2; //descrR
                // Create the dense matrices with the cusparse dense api
                CHECK_CUSPARSE(cusparseCreateDnMat(&matY2, B_nrows, (endObj - beginObj), B_nrows,
                                                  dY2, CUDA_R_64F, CUSPARSE_ORDER_COL))
                CHECK_CUSPARSE(cusparseCreateDnMat(&matX2, B_nrows, (endObj - beginObj), B_nrows,
                                                  dX2, CUDA_R_64F, CUSPARSE_ORDER_COL))

                // set the values of the dense matrix dC to vector dRPortion
                CHECK_CUSPARSE(cusparseDnMatSetValues(matY2, dRPortion))
                // copy the values of dRPortion whihc is an (endObj - beginOb) * A+ncols
                // in this computation we want to use the kernel formulation
                // C = alpha * op(A) . op(B) + beta * C 
                CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matX2, &beta, matY2, 
                        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeMM))

                CHECK_CUDA(cudaMalloc(&dBufferMM, bufferSizeMM));

                CHECK_CUSPARSE(cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                               &alpha, matA, matX2, &beta, matY2, CUDA_R_64F, 
                               CUSPARSE_SPMM_ALG_DEFAULT, &dBufferMM))

                CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matX2, &beta, matY2, CUDA_R_64F, 
                               CUSPARSE_SPMM_ALG_DEFAULT, &dBufferMM))

                /*iteration = 0;
                do {
                    // x = ri
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX, 1));
                    // initialise x' as ri too
                    if (iteration == 0) {
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dXPrime, 1));
                    }
                    // x = B.x' + ri where x = ri
                    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                    // x' = -1 * x + x'
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dXPrime, 1));
                    // max |x'|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd));
                    // get maxEps
                    CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    // x' = x
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dX, 1, dXPrime, 1));

                    //printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                    ++iteration;

                } while (maxEps > 1e-5 && iteration < maxIter);
                printf("___ VI PHASE TWO, OBJECTIVE %i, terminated at ITERATION %i\n", obj, iteration);
                // copy results
                thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + obj);
                */
                // clean up
                CHECK_CUSPARSE(cusparseDestroyDnMat(matX2))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matY2))
                CHECK_CUDA(cudaFree(dRPortion))
                CHECK_CUDA(cudaFree(dMaskTiled))
                CHECK_CUDA(cudaFree(dBufferMM))
                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                return EXIT_SUCCESS;
            }


            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::exit() {

                // destroy matrix/vector descriptors
                CHECK_CUSPARSE(cusparseDestroySpMat(matA))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecXPrime))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecRw))
                //CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                //CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
                //CHECK_CUSPARSE(cusparseDestroyDnMat(matD))
                CHECK_CUSPARSE(cusparseDestroy(handle))
                // device memory de-allocation
                CHECK_CUDA(cudaFree(dBuffer))
                CHECK_CUDA(cudaFree(dBufferC))
                CHECK_CUDA(cudaFree(dA_csrOffsets))
                CHECK_CUDA(cudaFree(dA_columns))
                CHECK_CUDA(cudaFree(dA_values))
                CHECK_CUDA(cudaFree(dA_rows_extra))
                CHECK_CUDA(cudaFree(dB_csrOffsets))
                CHECK_CUDA(cudaFree(dB_columns))
                CHECK_CUDA(cudaFree(dB_values))
                CHECK_CUDA(cudaFree(dB_rows_extra))
                CHECK_CUDA(cudaFree(dX))
                CHECK_CUDA(cudaFree(dXPrime))
                CHECK_CUDA(cudaFree(dY))
                CHECK_CUDA(cudaFree(dZ))
                CHECK_CUDA(cudaFree(dZPrime))
                CHECK_CUDA(cudaFree(dR))
                CHECK_CUDA(cudaFree(dRw))
                CHECK_CUDA(cudaFree(dRi))
                CHECK_CUDA(cudaFree(dW))
                CHECK_CUDA(cudaFree(dRowGroupIndices))
                CHECK_CUDA(cudaFree(dRow2RowGroupMapping))
                CHECK_CUDA(cudaFree(dMasking_nrows))
                CHECK_CUDA(cudaFree(dMasking_nnz))
                CHECK_CUDA(cudaFree(dResult))

                std::cout << ("____ CUDA EXIT ____\n");
                return EXIT_SUCCESS;
            }

            template class CudaValueIterationHandler<double>;
        }
    }
}


/*
{
    int k = 1000;
    std::vector<int> hRowGroupIndices(A_nrows);
    CHECK_CUDA(cudaMemcpy(hRowGroupIndices.data(), dRowGroupIndices, k * sizeof(int), cudaMemcpyDeviceToHost))
    printf("____ dRowGroupIndices: [");
    for (int i = 0; i < k; ++i) {
        std::cout << hRowGroupIndices[i] << " ";
    }
    std::cout << "...]\n";
}
 */