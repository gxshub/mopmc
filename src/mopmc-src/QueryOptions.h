//
// Created by guoxin on 21/12/23.
//

#ifndef MOPMC_QUERYOPTIONS_H
#define MOPMC_QUERYOPTIONS_H

#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include "mopmc-src/optimizers/BaseOptimizer.h"
#include "mopmc-src/solvers/BaseValueIteration.h"

namespace mopmc {

    struct QueryOptions {

        explicit QueryOptions() = default;

        enum { ACHIEVABILITY,
               CONVEX } QUERY_TYPE;
        enum { MSE,
               VAR } CONVEX_FUN;
        enum { CUDA_VI ,
               STANDARD_VI } VI;
        enum { CONSTRAINED,
               UNCONSTRAINED } CONSTRAINED_OPT;
    };
}// namespace mopmc

#endif//MOPMC_QUERYOPTIONS_H