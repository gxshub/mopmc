//
// Created by guoxin on 10/12/23.
//

#ifndef MOPMC_LINCOM_H
#define MOPMC_LINCOM_H

#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include <Eigen/Dense>
#include <memory>

namespace mopmc::optimization::auxiliary {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class LinearCombination {
    public:
        explicit LinearCombination(mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn);
        explicit LinearCombination(mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn,
                                   const std::vector<Vector<V>> &Points);

        static Vector<V> combine(const std::vector<Vector<V>> &Points, const Vector<V> &coeffs);
        V value(Vector<V> &x);
        Vector<V> gradient(Vector<V> &x);

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
        std::vector<Vector<V>> P;

    };

}
#endif //MOPMC_LINCOM_H
