//
// Created by guoxin on 2/02/24.
//

#include "ValueIteration.h"
#include "mopmc-src/auxiliary/Lincom.h"

namespace mopmc::value_iteration {

    template<typename V>
    ValueIterationHandler<V>::ValueIterationHandler(mopmc::QueryData<V, int> *queryData)
        : data(queryData),
          transitionMatrix(queryData->transitionMatrix),
          rowGroupIndices(queryData->rowGroupIndices),
          row2RowGroupMapping(queryData->row2RowGroupMapping),
          scheduler(queryData->scheduler),
          rewardVectors(queryData->rewardVectors),
          numRows(queryData->rowCount),
          numCols(queryData->colCount),
          numObjs(queryData->objectiveCount),
          iniRow(queryData->initialRow) {
        results.resize(numObjs + 1);
    }

    template<typename V>
    int ValueIterationHandler<V>::initialize() {
        P.resize(numCols, numCols);
        RR.resize(numObjs);
        for (uint64_t i = 0; i < numObjs; ++i) {
            RR[i] = VectorMap<V>(rewardVectors[i].data(), numRows);
        }
        return EXIT_SUCCESS;
    }

    template<typename V>
    int ValueIterationHandler<V>::valueIteration(const std::vector<V> &w) {
        this->valueIterationPhaseOne(w);
        this->valueIterationPhaseTwo();
        return EXIT_SUCCESS;
    }

    template<typename V>
    int ValueIterationHandler<V>::valueIterationPhaseOne(const std::vector<V> &w) {
        std::vector<V> w1 = w;
        Vector<V> w2 = VectorMap<V>(w1.data(), numObjs);
        Vector<V> R = mopmc::optimization::auxiliary::LinearCombination<V>::combine(RR, w2);
        Vector<V> X (numCols);
        Vector<V> Y = R;
        assert(R.size() == numRows);
        assert(Y.size() == numRows);
        const V tol = 1e-6;
        const uint64_t maxIter = 10000;
        uint64_t it = 0;
        while (it < maxIter) {
            Vector<V> X1(numCols);
            for (uint64_t i = 0; i < numCols; ++i) {
                uint64_t maxIdx = rowGroupIndices[i] + scheduler[i];
                X1(i) = Y(maxIdx);
                for (int j = 0; j < rowGroupIndices[i + 1] - rowGroupIndices[i]; ++j) {
                    if (j != scheduler[i] && X(i) < Y(rowGroupIndices[i] + j)) {
                        X1(i) = Y(rowGroupIndices[i] + j);
                        scheduler[i] = j;
                    }
                }
            }
            Y = transitionMatrix * X1 + R;
            if ((X1 - X).template lpNorm<Eigen::Infinity>() < tol) {
                X = std::move(X1);
                ++it;
                break;
            }
            X = std::move(X1);
            ++it;
        }
        results[numObjs] = X(iniRow);
        return EXIT_SUCCESS;
    }

    template<typename V>
    int ValueIterationHandler<V>::valueIterationPhaseTwo() {
        const V tol = 1e-6;
        const uint64_t maxIter = 10000;
        for (uint64_t k = 0; k < numObjs; ++k) {
            Vector<V> R(numCols);
            Eigen::SparseMatrix<V, Eigen::RowMajor> P_tmp (numCols, numCols);
            for (uint64_t i = 0; i < numCols; ++i) {
                const uint64_t j = rowGroupIndices[i] + scheduler[i];
                const uint64_t start = transitionMatrix.outerIndexPtr()[j];
                const uint64_t end = transitionMatrix.outerIndexPtr()[j + 1];
                for (uint64_t l = start; l < end; ++l) {
                    const uint64_t c = transitionMatrix.innerIndexPtr()[l];
                    P_tmp.insert(i, c) = transitionMatrix.valuePtr()[l];
                }
                R(i) = RR[k](j);
            }
            P_tmp.makeCompressed();
            P = std::move(P_tmp);
            Vector<V> Z(R);
            uint64_t it = 0;
            while (it < maxIter) {
                Vector<V> Z1 = P * Z + R;
                if ((Z1 - Z).template lpNorm<Eigen::Infinity>() < tol) {
                    Z = std::move(Z1);
                    ++it;
                    break;
                }
                Z = std::move(Z1);
                ++it;
            }
            results[k] = Z(iniRow);
        }
        return EXIT_SUCCESS;
    }

    template class ValueIterationHandler<double>;
}// namespace mopmc::value_iteration
