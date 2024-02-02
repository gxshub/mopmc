//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_LINOPT_H
#define MOPMC_LINOPT_H

#include "lp_lib.h"
#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <set>

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class LinOpt {
    public:
        int findOptimalSeparatingDirection(const std::vector<Vector<V>> &Vertices,
                                           const Vector<V> &gradient,
                                           const Vector<V> &sign,
                                           Vector<V> &weightVector,
                                           V &gap);

        int findOptimalProjectedDescentDirection(const std::vector<Vector<V>> &Directions,
                                                 const std::set<uint64_t> &exteriorIndices,
                                                 const Vector<V> &slope,
                                                 Vector<V> &descentDirection);

        int checkPointInConvexHull(const std::vector<Vector<V>> &Vertices,
                                   const Vector<V> &point,
                                   int &feasible);

        int findMaximumFeasibleStep(const std::vector<Vector<V>> &Vertices,
                                    const Vector<V> &gradient,
                                    Vector<V> point, V step);
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_LINOPT_H
