//
// Created by guoxin on 30/03/24.
//

#ifndef MOPMC_MINIMUMNORMPOINT_H
#define MOPMC_MINIMUMNORMPOINT_H

#include "BaseOptimizer.h"
#include "mopmc-src/auxiliary/LineSearch.h"
#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include <Eigen/Dense>
#include <set>
#include <vector>

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class MinimumNormPoint : public BaseOptimizer<V> {
    public:
        explicit MinimumNormPoint() = default;
        explicit MinimumNormPoint(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f) {}
        int optimizeSeparationDirection(Vector<V> &sepDirection,
                     Vector<V> &optimum,
                     V &margin,
                    const std::vector<Vector<V>> &Vertices,
                    const Vector<V> &pivot) override;
        Vector<V> alpha;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;
        std::set<uint64_t> activeVertices;
        int findNearestPointByDirection(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point,
                                        Vector<V> &weights);

        Vector<V> getVertexWeights() override {
            return alpha / alpha.template lpNorm<1>() ;
        }

    private:
        void initialize(const std::vector<Vector<V>> &Vertices);
        void performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices);
        bool checkSeparation(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point);
        V getSeparationMargin(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point);

        int64_t dimension{}, size{0};
        Vector<V> xCurrent, xNew, xNewTmp, dXCurrent;
    };

}// namespace mopmc::optimization::optimizers


#endif//MOPMC_MINIMUMNORMPOINT_H
