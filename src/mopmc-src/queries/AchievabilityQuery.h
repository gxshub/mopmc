//
// Created by guoxin on 2/11/23.
//

#ifndef MOPMC_ACHIEVABILITYQUERY_H
#define MOPMC_ACHIEVABILITYQUERY_H

#include "BaseQuery.h"
#include "mopmc-src/QueryData.h"
#include "mopmc-src/solvers/BaseValueIteration.h"
#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <thread>

namespace mopmc::queries {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename T, typename I>
    class AchievabilityQuery : public BaseQuery<T, I> {
    public:
        explicit AchievabilityQuery(const mopmc::QueryData<T, I> &data) : BaseQuery<T, I>(data){};
        explicit AchievabilityQuery(const mopmc::QueryData<T, I> &data, mopmc::value_iteration::BaseVIHandler<T> *VIHandler)
            : BaseQuery<T, I>(data, VIHandler){};

        void query() override;

        [[nodiscard]] uint64_t getMainLoopIterationCount() override {
            return this->iter;
        }

        [[nodiscard]] bool getResult() const {
            return this->achievable;
        }

        void printResult() override;

    private:
        std::vector<Vector<T>> VertexVectors, WeightVectors;
        uint64_t iter{};
        bool achievable{};
    };

}// namespace mopmc::queries

#endif//MOPMC_ACHIEVABILITYQUERY_H
