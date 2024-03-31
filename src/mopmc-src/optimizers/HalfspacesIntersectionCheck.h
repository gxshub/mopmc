//
// Created by guoxin on 30/03/24.
//

#ifndef MOPMC_HALFSPACESINTERSECTIONCHECK_H
#define MOPMC_HALFSPACESINTERSECTIONCHECK_H

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
    class HalfspacesIntersectionCheck {
    public:
        static int check(const std::vector<Vector<V>> &Points,
                         const std::vector<Vector<V>> &Directions,
                         Vector<V> &point,
                         bool &feasible);
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_HALFSPACESINTERSECTIONCHECK_H
