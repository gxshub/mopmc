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
        int minimize(Vector<V> &optimum,
                     const std::vector<Vector<V>> &Vertices,
                     const Vector<V> &pivot);
        Vector<V> alpha;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;
        std::set<uint64_t> activeVertices;

    private:
        void initialize(const std::vector<Vector<V>> &Vertices);
        void performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices);
        bool checkExit(const std::vector<Vector<V>> &Vertices);

        int64_t dimension{}, size{0};
        Vector<V> xCurrent, xNew, xNewTmp, dXCurrent;
        //std::set<uint64_t> nullVertices;
    };

}// namespace mopmc::optimization::optimizers


#endif//MOPMC_MINIMUMNORMPOINT_H
