//
// Created by guoxin on 26/11/23.
//

#ifndef MOPMC_LINESEARCH_H
#define MOPMC_LINESEARCH_H

#include <Eigen/Dense>

#include "mopmc-src/convex-functions/BaseConvexFunction.h"


namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class LineSearcher {
    public:

        explicit LineSearcher() = default;
        explicit LineSearcher(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f_);

        V findOptimalRelativeDistance(Vector<V> vLeft, Vector<V> vRight, V lambdaMax = static_cast<V>(1.));
        V findOptimalRelativeDistance_v0(Vector<V> vLeft, Vector<V> vRight, V lambdaMax = static_cast<V>(1.));

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *f_;
        Vector<V> vLeft_, vRight_;

    private:
        V g(V lambda);

        V dg(V lambda);
    };
}


#endif //MOPMC_LINESEARCH_H
