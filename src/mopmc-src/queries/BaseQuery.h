//
// Created by guoxin on 27/11/23.
//

#ifndef MOPMC_BASEQUERY_H
#define MOPMC_BASEQUERY_H

#include <storm/api/storm.h>
#include "../QueryData.h"
#include "../convex-functions/BaseConvexFunction.h"
#include "../optimizers/BaseOptimizer.h"
#include "../solvers/BaseValueIteration.h"

namespace mopmc::queries {

    template<typename V, typename I>
    class BaseQuery {
    public:

        explicit BaseQuery() = default;
        virtual ~BaseQuery() = default;
        explicit BaseQuery(const mopmc::QueryData<V,I> &data): queryData(data){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::value_iteration::BaseVIHandler<V> *valueIterSolver)
                           : queryData(data), VIhandler(valueIterSolver){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt): queryData(data), fn(f), innerOptimizer(priOpt), outerOptimizer(secOpt){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt,
                           mopmc::value_iteration::BaseVIHandler<V> *valueIterSolver)
            : queryData(data), fn(f), innerOptimizer(priOpt), outerOptimizer(secOpt), VIhandler(valueIterSolver){};

        virtual void query() = 0 ;
        virtual void printResult() {};
        virtual uint64_t getMainLoopIterationCount() {return 0;};

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
        mopmc::optimization::optimizers::BaseOptimizer<V> *innerOptimizer;
        mopmc::optimization::optimizers::BaseOptimizer<V> *outerOptimizer;
        mopmc::value_iteration::BaseVIHandler<V> *VIhandler;
        mopmc::QueryData<V, I> queryData;
    };



}

#endif //MOPMC_BASEQUERY_H
