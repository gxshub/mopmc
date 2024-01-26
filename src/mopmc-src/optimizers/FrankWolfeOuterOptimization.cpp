//
// Created by guoxin on 26/01/24.
//

#include "FrankWolfeOuterOptimization.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfeOuterOptimization<V>::minimize(Vector<V> &point,
                                                const std::vector<Vector<V>> &Vertices,
                                                const std::vector<Vector<V>> &Directions){
        size = Vertices.size();
        dimension = point.size();
        xNew = point;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 20;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slop = -1 * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                if (Directions[i].dot(point - Vertices[i]) < 1e-16) {
                    exteriorHSIndices.emplace_back(i);
                } else {
                    interiorHSIndices.emplace_back(i);
                }
            }
            if (exteriorHSIndices.empty()) {
                descentDirection = slop;
            } else if (exteriorHSIndices.size() == 1) {
                const Vector<V> &w = Directions[exteriorHSIndices[0]];
                descentDirection = slop - (slop.dot(w)) * slop;
            } else {
                this->linOpt.findOptimalProjectedDescentDirection(Directions, exteriorHSIndices, slop,
                                                                  descentDirection);
                for (auto i: exteriorHSIndices) {
                    if (Directions[i].dot(descentDirection) > 0) {
                        point = xNew;
                        return 0;
                    }
                }
            }
            V lambda = static_cast<V>(1000);
            for (auto i: interiorHSIndices) {
                const Vector<V> &w = Directions[i];
                if (w.dot(descentDirection) > 0) {
                    V lambda_x = w.dot(Vertices[i] - descentDirection) / (w.dot(descentDirection));
                    if (lambda > lambda_x) {
                        lambda = lambda_x;
                    }
                }
            }
            xNewTmp = xCurrent + lambda * descentDirection;
            lambda = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp);
            xNew = (1. - lambda) * xCurrent + lambda * xNewTmp;

            ++t;
        }
        point = xNew;
        return 0;
    }

    template class FrankWolfeOuterOptimization<double>;
}