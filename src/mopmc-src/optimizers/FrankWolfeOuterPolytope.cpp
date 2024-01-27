//
// Created by guoxin on 26/01/24.
//

#include "FrankWolfeOuterPolytope.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfeOuterPolytope<V>::minimize(Vector<V> &point,
                                             const std::vector<Vector<V>> &Vertices,
                                             const std::vector<Vector<V>> &Directions) {
        size = Vertices.size();
        dimension = point.size();
        xNew = point;
        std::set<uint64_t> exteriorHSIndices, interiorHSIndices;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 20;
        const V tol = 1e-12;
        bool exit = false;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slope = -1 * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                if (Directions[i].dot(Vertices[i] - xCurrent) < 1e-30) {
                    exteriorHSIndices.insert(i);
                } else {
                    interiorHSIndices.insert(i);
                }
            }
            if (exteriorHSIndices.empty()) {
                descentDirection = slope;
            } else if (exteriorHSIndices.size() == 1) {
                auto elem = exteriorHSIndices.begin();
                const Vector<V> &w = Directions[*elem];
                descentDirection = slope - (slope.dot(w)) * w;
            } else {
                this->linOpt.findOptimalProjectedDescentDirection(Directions, exteriorHSIndices, slope,
                                                                  descentDirection);
                for (auto i: exteriorHSIndices) {
                    if (Directions[i].dot(descentDirection) > 0) {
                        point = xNew;
                        exit = true;
                    }
                }
            }
            V lambda = static_cast<V>(1000);
            for (auto i: interiorHSIndices) {
                const Vector<V> &w = Directions[i];
                if (w.dot(descentDirection) > 0) {
                    V lambda_x = w.dot(Vertices[i] - xCurrent) / (w.dot(descentDirection));
                    if (lambda > lambda_x) {
                        lambda = lambda_x;
                    }
                }
            }
            xNewTmp = xCurrent + lambda * descentDirection;
            lambda = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp);
            xNew = (1. - lambda) * xCurrent + lambda * xNewTmp;
            ++t;
            if (exit || this->fn->value(xCurrent) - this->fn->value(xNew) < tol) { break; }
        }
        std::cout << "Outer optimization, FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        point = xNew;
        return 0;
    }

    template class FrankWolfeOuterPolytope<double>;
}// namespace mopmc::optimization::optimizers