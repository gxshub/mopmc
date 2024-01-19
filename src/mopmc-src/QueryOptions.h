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
               SE,
               VAR,
               SD } CONVEX_FUN;
        enum { BLENDED,
               BLENDED_STEP_OPT,
               AWAY_STEP,
               //LINOPT,
               SIMPLEX_GD,
               PGD } INNER_OPTIMIZER,
                OUTER_OPTIMIZER;
        enum { CUDA_VI } VI;
    };
}// namespace mopmc

#endif//MOPMC_QUERYOPTIONS_H
