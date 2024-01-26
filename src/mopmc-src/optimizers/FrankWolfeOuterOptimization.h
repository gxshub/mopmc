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
#include <vector>

#ifndef MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
#define MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class FrankWolfeOuterOptimization : public BaseOptimizer<V> {
    public:
        explicit FrankWolfeOuterOptimization() = default;
        explicit FrankWolfeOuterOptimization(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        int minimize(Vector<V> &point,
                     const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Directions) override;

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;

    private:
        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewTmp, dXCurrent;
        std::vector<uint64_t> interiorHSIndices, exteriorHSIndices;
    };
}


#endif//MOPMC_FRANKWOLFEOUTEROPTIMIZATION_H
