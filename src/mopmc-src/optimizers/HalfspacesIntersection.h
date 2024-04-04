//
// Created by guoxin on 30/03/24.
//

#ifndef MOPMC_HALFSPACESINTERSECTION_H
#define MOPMC_HALFSPACESINTERSECTION_H

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
    class HalfspacesIntersection {
    public:
        static bool findIntersectionPoint(const std::vector<Vector<V>> &Points,
                         const std::vector<Vector<V>> &Directions,
                         Vector<V> &point);
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_HALFSPACESINTERSECTION_H
