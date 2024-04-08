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
                                       const std::vector<Vector<V>> &BoundaryPoints,
                                       const std::vector<Vector<V>> &Directions) {

        interiorProjectionPhase(point, BoundaryPoints, Directions);
        //assert(this->fn->value(xNew) <= this->fn->value(point));
        exteriorProjectionPhase(point, BoundaryPoints, Directions);
        //if (!checkNonExteriorPoint(point, BoundaryPoints, Directions))
        //    throw std::runtime_error("Project gradient should return an non-exterior point");
        return EXIT_SUCCESS;
    }


    template<typename V>
    void ProjectedGradient<V>::interiorProjectionPhase(Vector<V> &point,
                                                       const std::vector<Vector<V>> &BoundaryPoints,
                                                       const std::vector<Vector<V>> &Directions) {
        /*
        {
            std::cout << "[interiorProjectionPhase] point as input: [";
            for (uint64_t i = 0; i < point.size(); ++i) {
                std::cout << point(i) << " ";
            }
            std::cout << "]\n";
        }
         */
        dimension = point.size();
        size = BoundaryPoints.size();
        xNew = point;
        std::set<uint64_t> boundaryIndices, nonboundaryIndices;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 100;
        const V tol = 1e-12;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slope = (-1.) * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                //if (Directions[i].dot(BoundaryPoints[i] - xCurrent) < 1e-30) {
                if (slope.dot(Directions[i]) > 0 && Directions[i].dot(BoundaryPoints[i] - xCurrent) < 1e-30) {
                    boundaryIndices.insert(i);
                } else {
                    nonboundaryIndices.insert(i);
                }
            }
            if (boundaryIndices.empty()) {
                descentDirection = slope;
            } else if (boundaryIndices.size() == 1) {
                auto elem = boundaryIndices.begin();
                const Vector<V> &w = Directions[*elem];
                const Vector<V> &r = BoundaryPoints[*elem];
                // projection of slope onto the hyperplane <x,w> = b
                descentDirection = halfspaceProjection(slope, r, w);
            } else {
                descentDirection = findProjectedDescentDirection(xCurrent, slope, BoundaryPoints, Directions, boundaryIndices);
            }
            // make a large enough range
            V lambda = static_cast<V>(1000);

            for (uint64_t i = 0; i < size; ++i) {
                const Vector<V> &w = Directions[i];
                if (w.dot(descentDirection) > 0) {
                    V lambda_x = w.dot(BoundaryPoints[i] - xCurrent) / (w.dot(descentDirection));
                    if (lambda > lambda_x) {
                        lambda = lambda_x;
                    }
                }
            }
            xNewTmp = xCurrent + lambda * descentDirection;
            lambda = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp);
            xNew = (1. - lambda) * xCurrent + lambda * xNewTmp;
            ++t;
            /*
            {
                std::cout << "[interiorProjectionPhase] updating outer point: [";
                for (uint64_t i = 0; i < point.size(); ++i) {
                    std::cout << xNew(i) << " ";
                }
                std::cout << "]\n";
            }
             */
            if ((xCurrent - xNew).template lpNorm<1>() < tol) {break;}
            //if (this->fn->value(xCurrent) - this->fn->value(xNew) < tol) { break; }
        }
        point = xNew;
        /*
        {
            std::cout << "[interiorProjectionPhase] point as output: [";
            for (uint64_t i = 0; i < point.size(); ++i) {
                std::cout << point(i) << " ";
            }
            std::cout << "]\n";
        }
         */
        std::cout << "[interiorProjectionPhase] finds minimum point at iteration: " << t << " (distance: " << this->fn->value(xNew) << ")\n";
    }

    template<typename V>
    void ProjectedGradient<V>::exteriorProjectionPhase(Vector<V> &point,
                                                       const std::vector<Vector<V>> &BoundaryPoints,
                                                       const std::vector<Vector<V>> &Directions) {
        /*
        {
            std::cout << "[exteriorProjectionPhase] point as input: [";
            for (uint64_t i = 0; i < point.size(); ++i) {
                std::cout << point(i) << " ";
            }
            std::cout << "]\n";
        }
         */

        const V step = 10.;
        const uint64_t maxIter = 100;
        const V tolerance = 1e-6;
        uint64_t iter = 0;
        Vector<V> tmpPoint0 = point, tmpPoint1, tmpPoint2;
        while (iter < maxIter) {
            Vector<V> slope = this->fn->subgradient(tmpPoint0) * (-1.);
            tmpPoint1 = tmpPoint0 + step * slope;
            const V lambda = this->lineSearcher.findOptimalRelativeDistance(tmpPoint0, tmpPoint1);
            tmpPoint1 = (1. - lambda) * tmpPoint0 + lambda * tmpPoint1;
            tmpPoint2 = dykstrasProjection(tmpPoint1, BoundaryPoints, Directions);
            if ((tmpPoint0 - tmpPoint2).template lpNorm<Eigen::Infinity>() < tolerance) {
                ++iter;
                break;
            }
            tmpPoint0 = tmpPoint2;
            ++iter;
            /*
            {
                std::cout << "[exteriorProjectionPhase] updating outer point: [";
                for (uint64_t i = 0; i < point.size(); ++i) {
                    std::cout << tmpPoint2(i) << " ";
                }
                std::cout << "]\n";
            }
             */

        }
        point = tmpPoint2;

        /*{
            std::cout << "[exteriorProjectionPhase] point as output: [";
            for (uint64_t i = 0; i < point.size(); ++i) {
                std::cout << point(i) << " ";
            }
            std::cout << "]\n";
        }
         */

    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::dykstrasProjection(const Vector<V> &point,
                                                       const std::vector<Vector<V>> &BoundaryPoints,
                                                       const std::vector<Vector<V>> &Directions,
                                                       const std::set<uint64_t> &indices) {

        const uint64_t m = BoundaryPoints[0].size();
        if (indices.empty()) {
            return point;
        }
        if (indices.size() == 1) {
            auto idx = indices.begin();
            return halfspaceProjection(point, BoundaryPoints[*idx], Directions[*idx]);
        }
        const auto d = indices.size();
        std::vector<Vector<V>> U(d + 1), Z(d);
        U[d] = point;
        for (int64_t i = 0; i < d; ++i) {
            U[i].resize(m);
            Z[i] = Vector<V>::Zero(m);
        }
        const uint64_t maxIter = 50;
        const V tolerance = 1e-15;
        uint_fast64_t it = 1;
        V tol;
        while (it < maxIter) {
            if ((U[0] - U[d]).template lpNorm<1>() < tolerance) {
                break;
            }
            U[0] = U[d];
            int64_t i = 0;
            for (auto idx: indices) {
                U[i + 1] = halfspaceProjection(U[i] + Z[i], BoundaryPoints[idx], Directions[idx]);
                Z[i] = U[i] + Z[i] - U[i + 1];
                i++;
            }
            ++it;
        }
        //std::cout << " - Dykstras projection, stops at " << it << "\n";
        Vector<V> point1 = U[d];
        V lambda = static_cast<V>(0.);
        for (auto idx: indices) {
            const Vector<V> &r = BoundaryPoints[idx];
            const Vector<V> &w = Directions[idx];
            if (w.dot(point - r) > 0) {
                lambda = std::max(lambda, w.dot(r - point) / w.dot(point1 - point));
            }
        }
        //std::cout << "[Projected gradient optimization - Dykstras Projection] iteration: " << it << "\n";
        point1 = point + lambda * (point1 - point);
        return point1;
    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::dykstrasProjection(const Vector<V> &point,
                                                       const std::vector<Vector<V>> &BoundaryPoints,
                                                       const std::vector<Vector<V>> &Directions) {
        std::set<uint64_t> indices;
        for (uint64_t i = 0; i < Directions.size(); ++i) {
            indices.insert(i);
        }
        return dykstrasProjection(point, BoundaryPoints, Directions, indices);
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
            V distance = direction.dot(point - boundaryPoint) /std::pow(direction.template lpNorm<2>(),2);
            return point - distance * direction;
        }
    }

    template<typename V>
    Vector<V> ProjectedGradient<V>::findProjectedDescentDirection(const Vector<V> &currentPoint,
                                                                  const Vector<V> &slope,
                                                                  const std::vector<Vector<V>> &BoundaryPoints,
                                                                  const std::vector<Vector<V>> &Directions,
                                                                  const std::set<uint64_t> &exteriorHSIndices) {
        const V gamma = 0.01;
        const Vector<V> newPoint = currentPoint + gamma * slope;
        Vector<V> projPoint = dykstrasProjection(newPoint, BoundaryPoints, Directions, exteriorHSIndices);
        Vector<V> projSlope = projPoint - currentPoint;
        if (projSlope.dot(slope) > 0) {
            return projSlope;
        } else {
            return Vector<V>::Zero(dimension);
        }
    }

    template<typename V>
    bool ProjectedGradient<V>::checkNonExteriorPoint(Vector<V> &point,
                                                     const std::vector<Vector<V>> &BoundaryPoints,
                                                     const std::vector<Vector<V>> &Directions) {
        const V roundingError = 1e-12;
        bool nonExterior = true;
        const uint64_t maxIter = Directions.size();
        uint64_t i = 0;
        while (i < maxIter) {
            if (Directions[i].dot(point) > Directions[i].dot(BoundaryPoints[i]) + roundingError) {
                nonExterior = false;
                break;
            }
            ++i;
        }
        if (!nonExterior) {
            std::cout << "Directions.size(): " << Directions.size()<<"\n";
            std::cout << "[Project gradient] exterior point - Directions[i].dot(point) - Directions[i].dot(BoundaryPoints[i]): "
                      << Directions[i].dot(point) - Directions[i].dot(BoundaryPoints[i]) <<"\n";
        }
        return nonExterior;
    }

    template class ProjectedGradient<double>;
}// namespace mopmc::optimization::optimizers