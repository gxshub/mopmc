//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include "mopmc-src/QueryData.h"
#include "mopmc-src/queries/BaseQuery.h"
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include <storm/storage/SparseMatrix.h>

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V, typename I>
    class ConvexQuery : public BaseQuery<V, I>{
    public:
        explicit ConvexQuery(const mopmc::QueryData<V, I> &data) : BaseQuery<V, I>(data) {};
        ConvexQuery(const mopmc::QueryData<V,I> &data,
                       mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                       mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                       mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt)
                       : BaseQuery<V, I>(data, f, priOpt, secOpt) {};
        ConvexQuery(const mopmc::QueryData<V,I> &data,
                    mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt,
                    mopmc::value_iteration::BaseVIHandler<V> *valueIteration)
                : BaseQuery<V, I>(data, f, priOpt, secOpt, valueIteration) {};

        void query() override;

        Eigen::SparseMatrix<V, Eigen::RowMajor> P_;
    };
}

#endif //MOPMC_CONVEXXQUERY_H
