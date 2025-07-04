//
// Created by guoxin on 26/01/24.
//

#ifndef MOPMC_PROJECTEDGRADIENT_H
#define MOPMC_PROJECTEDGRADIENT_H

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
    class ProjectedGradient : public BaseOptimizer<V> {
    public:
        explicit ProjectedGradient() = default;
        explicit ProjectedGradient(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f) {
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
        void interiorProjectionPhase(Vector<V> &point,
                                     const std::vector<Vector<V>> &BoundaryPoints,
                                     const std::vector<Vector<V>> &Directions);

        void exteriorProjectionPhase(Vector<V> &point,
                                     const std::vector<Vector<V>> &BoundaryPoints,
                                     const std::vector<Vector<V>> &Directions);

        Vector<V> dykstrasProjection(const Vector<V> &point,
                                     const std::vector<Vector<V>> &BoundaryPoints,
                                     const std::vector<Vector<V>> &Directions,
                                     const std::set<uint64_t> &indices);

        Vector<V> dykstrasProjection(const Vector<V> &point,
                                     const std::vector<Vector<V>> &BoundaryPoints,
                                     const std::vector<Vector<V>> &Directions);

        Vector<V> halfspaceProjection(const Vector<V> &point,
                                      const Vector<V> &boundaryPoint,
                                      const Vector<V> &direction);

        Vector<V> findProjectedDescentDirection(const Vector<V> &currentPoint,
                                                const Vector<V> &slope,
                                                const std::vector<Vector<V>> &BoundaryPoints,
                                                const std::vector<Vector<V>> &Directions,
                                                const std::set<uint64_t> &indices);

        //int64_t dimension{}, size{};
        //Vector<V> xCurrent, xNew, xNewTmp;
    };
}// namespace mopmc::optimization::optimizers


#endif//MOPMC_PROJECTEDGRADIENT_H
