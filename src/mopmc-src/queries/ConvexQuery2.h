//
// Created by guoxin on 16/01/24.
//

#ifndef MOPMC_CONVEXQUERY2_H
#define MOPMC_CONVEXQUERY2_H
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include "BaseQuery.h"
#include "../QueryData.h"

namespace mopmc::queries {
    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V, typename I>
    class ConvexQuery2 : public BaseQuery<V, I> {
    public:
        ConvexQuery2(const mopmc::QueryData<V,I> &data,
                    mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt,
                    mopmc::value_iteration::BaseVIHandler<V> *valueIteration)
            : BaseQuery<V, I>(data, f, priOpt, secOpt, valueIteration) {};

        void query() override;
    };
}


#endif//MOPMC_CONVEXQUERY2_H
