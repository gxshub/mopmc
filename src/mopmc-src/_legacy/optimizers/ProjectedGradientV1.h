//
// Created by guoxin on 26/01/24.
//

#ifndef MOPMC_PROJECTEDGRADIENTV1_H
#define MOPMC_PROJECTEDGRADIENTV1_H

#include "mopmc-src/optimizers/BaseOptimizer.h"
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
    class ProjectedGradientV1 : public BaseOptimizer<V> {
    public:
        explicit ProjectedGradientV1() = default;
        explicit ProjectedGradientV1(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f) {
            this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
        };

        int minimize(Vector<V> &point,
                     const std::vector<Vector<V>> &BoundaryPoints,
                     const std::vector<Vector<V>> &Directions) override;

        bool checkNonExteriorPoint(Vector<V> &point,
                                   const std::vector<Vector<V>> &BoundaryPoints,
                                   const std::vector<Vector<V>> &Directions);

        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;

    private:
        void exteriorProjectionPhase();

        Vector<V> dykstrasProjection(const Vector<V> &point,
                                     const std::vector<Vector<V>> &BoundaryPoints,
                                     const std::vector<Vector<V>> &Directions,
                                     const std::set<uint64_t> &indices);

        Vector<V> halfspaceProjection(const Vector<V> &point,
                                      const Vector<V> &boundaryPoint,
                                      const Vector<V> &direction);

        Vector<V> findProjectedDescentDirection(const Vector<V> &point,
                                                const Vector<V> &slope,
                                                const std::vector<Vector<V>> &BoundaryPoints,
                                                const std::vector<Vector<V>> &Directions,
                                                const std::set<uint64_t> &indices);

        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewTmp;
    };
}// namespace mopmc::optimization::optimizers


#endif//MOPMC_PROJECTEDGRADIENTV1_H
