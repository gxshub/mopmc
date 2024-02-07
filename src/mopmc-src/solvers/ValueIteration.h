//
// Created by guoxin on 2/02/24.
//

#ifndef MOPMC_VALUEITERATION_H
#define MOPMC_VALUEITERATION_H

#include "BaseValueIteration.h"
#include "mopmc-src/QueryData.h"

namespace mopmc::value_iteration {
    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class ValueIterationHandler : public mopmc::value_iteration::BaseVIHandler<V> {
    public:
        explicit ValueIterationHandler(mopmc::QueryData<V, int> *queryData);
        int initialize() override;
        //int exit() override;
        //int valueIteration(const std::vector<double> &w) override;
        [[nodiscard]] const std::vector<V> &getResults() const override {
            return results;
        }
        int valueIteration(const std::vector<V> &w) override;
        int valueIterationPhaseOne(const std::vector<V> &w);
        int valueIterationPhaseTwo();

    private:
        mopmc::QueryData<V, int> *data;
        Eigen::SparseMatrix<V, Eigen::RowMajor> transitionMatrix;
        std::vector<std::vector<V>> rewardVectors;
        std::vector<int> scheduler;
        std::vector<int> rowGroupIndices;
        std::vector<int> row2RowGroupMapping;
        std::vector<V> weightedRewardVector;
        std::vector<V> results;
        uint64_t numRows{}, numCols{}, numObjs{};
        uint64_t iniRow{};
        std::vector<Vector<V>> RR;
        Eigen::SparseMatrix<V, Eigen::RowMajor> P;
    };
}


#endif//MOPMC_VALUEITERATION_H
