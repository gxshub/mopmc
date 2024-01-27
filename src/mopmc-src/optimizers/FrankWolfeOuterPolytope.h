//
// Created by guoxin on 26/01/24.
//

#include "BaseOptimizer.h"
#include "LinOpt.h"
#include "LineSearch.h"
#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

#ifndef MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
#define MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class FrankWolfeOuterPolytope : public BaseOptimizer<V> {
    public:
        explicit FrankWolfeOuterPolytope() = default;
        explicit FrankWolfeOuterPolytope(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f) {
            this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
        };

        int minimize(Vector<V> &point,
                     const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Directions) override;

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;

    private:
        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewTmp;
    };
}// namespace mopmc::optimization::optimizers


#endif//MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
