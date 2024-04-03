//
// Created by guoxin on 2/02/24.
//

#ifndef MOPMC_SEPERATIONDIRECTIONOPTIMIZER_H
#define MOPMC_SEPERATIONDIRECTIONOPTIMIZER_H

#include "BaseOptimizer.h"
#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class SeparationHyperplaneOptimizer {
    public:
        explicit SeparationHyperplaneOptimizer() = default;
        ~SeparationHyperplaneOptimizer() = default;

        int findMaximumSeparatingDirection(const std::vector<Vector<V>> &Vertices,
                                           const Vector<V> &threshold,
                                           const Vector<V> &sign,
                                           Vector<V> &direction,
                                           V &distance);
    };

}// namespace mopmc::optimization::optimizers

#endif//MOPMC_SEPERATIONDIRECTIONOPTIMIZER_H
