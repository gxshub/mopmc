//
// Created by guoxin on 2/11/23.
//

#ifndef MOPMC_ACHIEVABILITYQUERY_H
#define MOPMC_ACHIEVABILITYQUERY_H

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <thread>
#include "BaseQuery.h"
#include "../QueryData.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../optimizers/LinOpt.h"
#include "../optimizers/PolytopeTypeEnum.h"
#include "../solvers/BaseValueIteration.h"

namespace mopmc::queries {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename T, typename I>
    class AchievabilityQuery : public BaseQuery<T, I> {
    public:
        explicit AchievabilityQuery(const mopmc::QueryData<T, I> &data) : BaseQuery<T, I>(data) {};
        explicit AchievabilityQuery(const mopmc::QueryData<T, I> &data, mopmc::value_iteration::BaseVIHandler<T> *VIHandler)
            : BaseQuery<T, I>(data, VIHandler) {};

        void query() override;
    };
}

#endif //MOPMC_ACHIEVABILITYQUERY_H
