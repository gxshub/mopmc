//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFE_H
#define MOPMC_FRANKWOLFE_H

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

    enum FWOption {
        SIMPLEX_GD,
        AWAY_STEP,
        BLENDED,
        BLENDED_STEP_OPT
    };

    template<typename V>
    class FrankWolfe : public BaseOptimizer<V> {
    public:
        explicit FrankWolfe() = default;
        explicit FrankWolfe(FWOption optMethod, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);
        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;
        FWOption fwOption{};
        Vector<V> alpha;
        std::set<uint64_t> activeVertices;

    private:
        [[deprecated]] Vector<V> argmin(const std::vector<Vector<V>> &Vertices);
        [[deprecated]] void initialize(const std::vector<Vector<V>> &Vertices, V &delta, const V &scale);
        void initialize(const std::vector<Vector<V>> &Vertices);
        void performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices);
        void forwardOrAwayStepUpdate(uint64_t &fwdInd, Vector<V> &fwdVec,
                                     uint64_t &awyInd, Vector<V> &awyVec,
                                     V &gamma, V &gammaMax, bool &isFwd);
        void performForwardOrAwayStepDescent(const std::vector<Vector<V>> &Vertices);
        void checkForwardStep(const std::vector<Vector<V>> &Vertices, uint64_t &fwdInd, Vector<V> &fwdVec, V &fwdEps);
        void checkAwayStep(const std::vector<Vector<V>> &Vertices, uint64_t &awyInd, Vector<V> &awyVec, V &awyEps);
        bool checkExit(const std::vector<Vector<V>> &Vertices);

        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewTmp, dXCurrent;
        std::set<uint64_t> nullVertices;
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_FRANKWOLFE_H
