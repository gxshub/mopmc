//
// Created by guoxin on 26/01/24.
//

#ifndef MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
#define MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H

#include "BaseOptimizer.h"
#include "mopmc-src/auxiliary/LineSearch.h"
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
    class FrankWolfeOuterOptimizer : public BaseOptimizer<V> {
    public:
        explicit FrankWolfeOuterOptimizer() = default;
        explicit FrankWolfeOuterOptimizer(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f) {
            this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
        };

        int minimize(Vector<V> &point,
                     const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Directions) override;

        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;

    private:
        int findOptimalProjectedDescentDirection(const std::vector<Vector<V>> &Directions,
                                                 const std::set<uint64_t> &exteriorIndices,
                                                 const Vector<V> &slope,
                                                 Vector<V> &descentDirection);
        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewTmp;
    };
}// namespace mopmc::optimization::optimizers


#endif//MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
