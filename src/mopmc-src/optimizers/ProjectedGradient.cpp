//
//Created by guoxin on 26/01/24.
//

#include "ProjectedGradient.h"
#include "HalfspacesIntersection.h"
#include "lp_lib.h"
#include <iostream>

        namespace mopmc::optimization::optimizers {

    template<typename V>
    int ProjectedGradient<V>::minimize(Vector<V> &point,
                                       const std::vector<Vector<V>> &Vertices,
                                       const std::vector<Vector<V>> &Directions) {

        /*
        if (!checkNonExteriorPoint(point, Vertices, Directions)) {
            throw std::runtime_error("Project gradient requires a non-exterior initial point");
        }
         */
        dimension = point.size();
        size = Vertices.size();
        xNew = point;
        std::set<uint64_t> boundaryIndices, nonboundaryIndices;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 20;
        const V tol = 1e-12;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slope = -1 * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                if (slope.dot(Directions[i]) > 0 && Directions[i].dot(Vertices[i] - xCurrent) < 1e-30) {
                    boundaryIndices.insert(i);
                } else {
                    nonboundaryIndices.insert(i);
                }
            }
            if (boundaryIndices.empty()) {
                descentDirection = slope;
            }
            else if (boundaryIndices.size() == 1) {
                auto elem = boundaryIndices.begin();
                const Vector<V> &w = Directions[*elem];
                const Vector<V> &r = Vertices[*elem];
                // projection of slope onto the hyperplane <x,w> = b
                descentDirection = halfspaceProjection(slope, r, w);
            }
            else {
                descentDirection = findProjectedDescentDirection(xCurrent, slope, Vertices, Directions, boundaryIndices);
            }
            // make a large enough range
            V lambda = static_cast<V>(1000);
            for (auto i: nonboundaryIndices) {
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
        std::cout << "[Projected gradient optimization] finds minimum point at iteration: " << t << " (distance: " << this->fn->value(xNew) << ")\n";
        //assert(this->fn->value(xNew) <= this->fn->value(point));
        point = xNew;
        /*
        if (!HalfspacesIntersection<V>::checkNonExteriorPoint(point, Vertices, Directions)) {
            throw std::runtime_error("Project gradient should return an non-exterior point");
        }
         */
        return EXIT_SUCCESS;
    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::dykstrasProjection(const Vector<V> &point,
                                                       const std::vector<Vector<V>> &Vertices,
                                                       const std::vector<Vector<V>> &Directions,
                                                       const std::set<uint64_t> &indices) {

        const uint64_t m = Vertices[0].size();
        if (indices.empty()) {
            return point;
        }
        if (indices.size() == 1) {
            auto idx = indices.begin();
            return halfspaceProjection(point, Vertices[*idx], Directions[*idx]);
        }
        const auto d = indices.size();
        std::vector<Vector<V>> U(d + 1), Z(d);
        U[d] = point;
        for (int64_t i = 0; i < d; ++i) {
            U[i].resize(m);
            Z[i] = Vector<V>::Zero(m);
        }
        const uint64_t maxIter = 200;
        const V tolerance = 1e-24;
        uint_fast64_t it = 1;
        V tol;
        while (it < maxIter) {
            if ((U[0] - U[d]).template lpNorm<1>() < tolerance) {
                break;
            }
            U[0] = U[d];
            int64_t i = 0;
            for (auto idx: indices) {
                U[i + 1] = halfspaceProjection(U[i] + Z[i], Vertices[idx], Directions[idx]);
                Z[i] = U[i] + Z[i] - U[i + 1];
                i++;
            }
            ++it;
        }
        //std::cout << " - Dykstras projection, stops at " << it << "\n";
        return U[d];
    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::halfspaceProjection(const Vector<V> &point,
                                                        const Vector<V> &boundaryPoint,
                                                        const Vector<V> &direction) {
        //assert(direction.size() == boundaryPoint.size());
        //assert(point.size() == boundaryPoint.size());
        if (direction.dot(point - boundaryPoint) <= 0) {
            return point;
        } else {
            V distance = direction.dot(point - boundaryPoint);// note. std::pow(direction.template lpNorm<2>(),2) = 1
            return point - distance * direction;
        }
    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::findProjectedDescentDirection(const Vector<V> &point,
                                                                  const Vector<V> &slope,
                                                                  const std::vector<Vector<V>> &Vertices,
                                                                  const std::vector<Vector<V>> &Directions,
                                                                  const std::set<uint64_t> &exteriorHSIndices) {
        const V gamma = 0.01;
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
        Vector<V> projSlope = outPoint + lambda * (projPoint - outPoint) - point;
        if (projSlope.dot(slope) > 0) {
            return projSlope;
        } else {
            return slope;
        }
    }

    template<typename V>
    bool ProjectedGradient<V>::checkNonExteriorPoint(Vector<V> &point,
                                                     const std::vector<Vector<V>> &Vertices,
                                                     const std::vector<Vector<V>> &Directions) {
        bool nonExterior = true;
        for (uint64_t i = 0; i < Directions.size(); ++i) {
            if (Directions[i].dot(point) > Directions[i].dot(Vertices[i])) {
                nonExterior = false;
                break;
            }
        }
        return nonExterior;
    }

    template<typename V>
    void ProjectedGradient<V>::exteriorProjectionPhase(){
        const V step = 0.0001;
        xCurrent = xNew;
        Vector<V> gradient = this->fn->subgradient(xCurrent);
        xNew = xCurrent - step * gradient;
    }

    template class ProjectedGradient<double>;
}// namespace mopmc::optimization::optimizers