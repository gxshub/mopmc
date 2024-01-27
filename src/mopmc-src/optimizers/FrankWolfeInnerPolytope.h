//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFEINNERPOLYTOPE_H
#define MOPMC_FRANKWOLFEINNERPOLYTOPE_H

#include "BaseOptimizer.h"
#include "LinOpt.h"
#include "LineSearch.h"
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

    enum FWOption { SIMPLEX_GD, AWAY_STEP };

    template<typename V>
    class FrankWolfeInnerPolytope : public BaseOptimizer<V> {
    public:
        explicit FrankWolfeInnerPolytope() = default;
        explicit FrankWolfeInnerPolytope(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                                         FWOption optMethod=FWOption::SIMPLEX_GD) : BaseOptimizer<V>(f), fwOption(optMethod) {
            this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
        }
        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;
        FWOption fwOption{};
        Vector<V> alpha;
        std::set<uint64_t> activeVertices;

    private:
        void initialize(const std::vector<Vector<V>> &Vertices);
        void performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices);
        void forwardOrAwayStepUpdate(uint64_t &fwdInd, Vector<V> &fwdVec,
                                     uint64_t &awyInd, Vector<V> &awyVec,
                                     V &gamma, V &gammaMax, bool &isFwd);
        void performForwardOrAwayStepDescent(const std::vector<Vector<V>> &Vertices);
        void checkForwardStep(const std::vector<Vector<V>> &Vertices, uint64_t &fwdInd, Vector<V> &fwdVec, V &fwdEps);
        void checkAwayStep(const std::vector<Vector<V>> &Vertices, uint64_t &awyInd, Vector<V> &awyVec, V &awyEps);
        bool checkExit(const std::vector<Vector<V>> &Vertices);

        int64_t dimension{}, size{0};
        Vector<V> xCurrent, xNew, xNewTmp, dXCurrent;
        std::set<uint64_t> nullVertices;
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_FRANKWOLFEINNERPOLYTOPE_H
