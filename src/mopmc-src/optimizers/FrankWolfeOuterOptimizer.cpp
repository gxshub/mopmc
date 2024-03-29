//
// Created by guoxin on 26/01/24.
//

#include "FrankWolfeOuterOptimizer.h"
#include "lp_lib.h"
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfeOuterOptimizer<V>::minimize(Vector<V> &point,
                                              const std::vector<Vector<V>> &Vertices,
                                              const std::vector<Vector<V>> &Directions) {

        dimension = point.size();
        size = Vertices.size();
        xNew = point;
        std::set<uint64_t> exteriorHSIndices, interiorHSIndices;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 20;
        const V tol = 1e-12;
        //bool exit = false;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slope = -1 * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                if (slope.dot(Directions[i]) > 0 && Directions[i].dot(Vertices[i] - xCurrent) < 1e-30) {
                    exteriorHSIndices.insert(i);
                } else {
                    interiorHSIndices.insert(i);
                }
            }
            std::cout << "exteriorHSIndices.size(): " << exteriorHSIndices.size() <<"\n";
            if (exteriorHSIndices.empty()) {
                descentDirection = slope;
            } else if (exteriorHSIndices.size() == 1) {
                auto elem = exteriorHSIndices.begin();
                const Vector<V> &w = Directions[*elem];
                // projection of slope onto the hyperplane <x,w> = b
                descentDirection = slope - (slope.dot(w)) * w;
            } else {
                descentDirection = findProjectedDescentDirection(xCurrent, slope, Vertices, Directions, exteriorHSIndices);
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
            if (this->fn->value(xCurrent) - this->fn->value(xNew) < tol) { break; }
        }
        std::cout << "Outer optimization, FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        //assert(this->fn->value(xNew) <= this->fn->value(point));
        point = xNew;
        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfeOuterOptimizer<V>::dykstrasProjection(const Vector<V> &point,
                                                              const std::vector<Vector<V>> &Vertices,
                                                              const std::vector<Vector<V>> &Directions,
                                                              const std::set<uint64_t> &exteriorHSIndices) {

        const uint64_t m = Vertices[0].size();
        if (exteriorHSIndices.empty()) {
            return point;
        }
        if (exteriorHSIndices.size() == 1) {
            auto idx = exteriorHSIndices.begin();
            return projectFromPointToHalfspace(point, Vertices[*idx], Directions[*idx]);
        }
        const auto d = exteriorHSIndices.size();
        std::vector<Vector<V>> U(d + 1), Z(d);
        U[d] = point;
        for (int64_t i = 0; i < d; ++i) {
            U[i].resize(m);
            Z[i] = Vector<V>::Zero(m);
        }
        const uint64_t maxIter = 200;
        const V tolerance = 1e-5;
        uint_fast64_t it = 1;
        V tol;
        while (it < maxIter) {
            if ((U[0] - U[d]).template lpNorm<1>() < tolerance) {
                break;
            }
            U[0] = U[d];
            int64_t i = 0;
            for (auto idx: exteriorHSIndices) {
                U[i + 1] = projectFromPointToHalfspace(U[i] + Z[i], Vertices[idx], Directions[idx]);
                Z[i] = U[i] + Z[i] - U[i + 1];
                i++;
            }
            ++it;
        }
        //std::cout << "Dykstras projection, stops at " << it << "\n";
        return U[d];
    }

    template<typename V>
    Vector<V> FrankWolfeOuterOptimizer<V>::projectFromPointToHalfspace(const Vector<V> &point,
                                                                       const Vector<V> &vertex,
                                                                       const Vector<V> &direction) {
        assert(direction.size() == vertex.size());
        assert(point.size() == vertex.size());
        if (direction.dot(point - vertex) <= 0) {
            return point;
        } else {
            V distance = direction.dot(point - vertex) / std::pow(direction.template lpNorm<2>(), 2);
            return point - distance * direction;
        }
    }

    template<typename V>
    Vector<V> FrankWolfeOuterOptimizer<V>::findProjectedDescentDirection(const Vector<V> &point,
                                                                         const Vector<V> &slope,
                                                                         const std::vector<Vector<V>> &Vertices,
                                                                         const std::vector<Vector<V>> &Directions,
                                                                         const std::set<uint64_t> &exteriorHSIndices) {
        const V gamma = 0.1;
        const Vector<V> outPoint = point + gamma * slope;
        Vector<V> projPoint = dykstrasProjection(outPoint, Vertices, Directions, exteriorHSIndices);
        V lambda = static_cast<V>(1.);
        for (auto idx: exteriorHSIndices) {
            const Vector<V> &w = Vertices[idx];
            const Vector<V> &r = Directions[idx];
            if (w.dot(projPoint - r) > 0) {
                lambda = std::max(lambda, w.dot(r - outPoint) / w.dot(projPoint - outPoint));
            }
        }
        const Vector<V> &projSlope = outPoint + lambda * (projPoint - outPoint) - point;
        if (projSlope.dot(slope) > 0) {
            return projSlope;
        } else {
            return slope;
        }
    }

    template class FrankWolfeOuterOptimizer<double>;
}// namespace mopmc::optimization::optimizers