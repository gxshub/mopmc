//
// Created by guoxin on 4/12/23.
//

#include "ProjectedGradientDescent.h"
#include "mopmc-src/auxiliary/Lincom.h"
#include "mopmc-src/optimizers/LineSearch.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int ProjectedGradientDescent<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) {
        assert(this->projectionType == ProjectionType::UnitSimplex);
        assert(!Vertices.empty());
        if (Vertices.size() == 1) {
            this->alpha.resize(1);
            this->alpha(0) = static_cast<V>(1.);
        } else {
            assert(this->alpha.size() == Vertices.size() - 1);
            this->alpha.resize(Vertices.size());
            this->alpha(alpha.size() - 1) = static_cast<V>(0.);
        }
        this->optimalPoint = argminUnitSimplexProjection(this->alpha, Vertices);
        point = this->optimalPoint;
        return 0;
    }

    template<typename V>
    int ProjectedGradientDescent<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices,
                                              const std::vector<Vector<V>> &Directions) {
        assert(!Vertices.empty() && Vertices.size() == Directions.size());
        this->optimalPoint = minimizeByAdaptiveStepSize(point, Vertices, Directions);
        //this->optimalPoint = minimizeByFixedStepSize(point, Vertices, Directions);
        point = this->optimalPoint;
        return 0;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::minimizeByAdaptiveStepSize(const Vector<V> &point,
                                                                      const std::vector<Vector<V>> &Vertices,
                                                                      const std::vector<Vector<V>> &Directions) {

        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher(this->fn);
        const uint64_t maxIter = 2000;
        const V beta = static_cast<V>(0.8);
        const V epsilon = static_cast<V>(1.e-6);
        const uint64_t dim = point.size();
        V gamma0(static_cast<V>(1e-2)), gamma1(static_cast<V>(1.));
        Vector<V> xCurrent(point), xNew(dim), xTemp(dim), xTemp1(dim), xGrad(dim);
        uint_fast64_t it = 0;
        while (it < maxIter) {
            xGrad = this->fn->subgradient(xCurrent);
            gamma0 = static_cast<V>(1.);
            V g = xGrad.template lpNorm<2>();
            while (this->fn->value(xCurrent - xGrad) > this->fn->value(xCurrent) - gamma0 * 0.5 * std::pow(g, 2)) {
                gamma0 *= beta;
                if (gamma0 < 1e-2) { break; }
            }
            xTemp = xCurrent - gamma0 * this->fn->subgradient(xCurrent);
            xTemp1 = dykstrasProjection(xTemp, Vertices, Directions);
            gamma1 = lineSearcher.findOptimalRelativeDistance(xCurrent, xTemp1, 1.);
            xNew = (1. - gamma1) * xCurrent + gamma1 * xTemp1;
            //assert(this->fn->value(xNew) <= this->fn->value(xCurrent));
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                ++it;
                std::cout << "Projected GD exits due to small tolerance (" << error << ")\n";
                break;
            }
            xCurrent = xNew;
            ++it;
        }
        std::cout << "*Projected GD* stops at iteration: " << it << ", nearest distance: " << this->fn->value(xNew) << "\n";
        return xNew;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::minimizeByFixedStepSize(const Vector<V> &point,
                                                                   const std::vector<Vector<V>> &Vertices,
                                                                   const std::vector<Vector<V>> &Directions) {

        const uint64_t maxIter = 10000;
        const V epsilon = static_cast<V>(1.e-8);
        const V gamma = static_cast<V>(0.001);

        const uint64_t dim = point.size();
        Vector<V> xCurrent = point, xNew(dim), xTemp(dim);
        Vector<V> grad(dim);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            grad = this->fn->subgradient(xCurrent);
            xTemp = xCurrent - gamma * grad;
            xNew = projectToHalfSpaces(xTemp, Vertices, Directions);
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                break;
            }
            xCurrent = xNew;
        }
        std::cout << "*Projected GD* stops at iteration: " << it << ", near distance: " << this->fn->value(xNew) << "\n";
        return xNew;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::projectFromPointToHyperPlane(const Vector<V> &point, const Vector<V> &vertex, const Vector<V> &direction) {
        assert(direction.size() == vertex.size());
        assert(point.size() == vertex.size());
        V distance = direction.dot(point - vertex) / std::pow(direction.template lpNorm<2>(), 2);
        return point - distance * direction;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::intersectLineWithHalfSpaces(const Vector<V> &point1,
                                                                       const Vector<V> &point2,
                                                                       const std::vector<Vector<V>> &Vertices,
                                                                       const std::vector<Vector<V>> &Directions) {
        if ((point1 - point2).template lpNorm<1>() < 1e-12) {
            return point2;
        }
        V lambda = static_cast<V>(1.);
        for (uint_fast64_t i = 0; i < Vertices.size(); ++i) {
            V lambda1 = Directions[i].dot(Vertices[i] - point2) / (Directions[i].dot(point1 - point2));
            if (lambda < lambda1) {
                lambda = lambda1;
            }
        }
        return lambda * point1 + (1. - lambda) * point2;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::dykstrasProjection(const Vector<V> &point,
                                                              const std::vector<Vector<V>> &Vertices,
                                                              const std::vector<Vector<V>> &Directions) {

        const uint64_t m = Vertices[0].size();
        std::vector<int64_t> pointIndices;
        for (int64_t i = 0; i < Vertices.size(); ++i) {
            if (Directions[i].dot(point - Vertices[i]) > 0.) {
                pointIndices.emplace_back(i);
            }
        }
        if (pointIndices.empty()) {
            return point;
        }
        if (pointIndices.size() == 1) {
            int64_t idx = pointIndices[0];
            return projectFromPointToHyperPlane(point, Vertices[idx], Directions[idx]);
        }
        const auto d = pointIndices.size();
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
            tol = (U[0] - U[d]).template lpNorm<1>();
            if ((U[0] - U[d]).template lpNorm<1>() < tolerance) {
                break;
            }
            U[0] = U[d];
            for (int64_t i = 0; i < d; ++i) {
                auto idx1 = pointIndices[i];
                U[i + 1] = projectFromPointToHyperPlane(U[i] + Z[i], Vertices[idx1], Directions[idx1]);
                Z[i] = U[i] + Z[i] - U[i + 1];
            }
            ++it;
        }
        return U[d];
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::projectToHalfSpaces(const Vector<V> &point,
                                                               const std::vector<Vector<V>> &Vertices,
                                                               const std::vector<Vector<V>> &Directions) {
        uint64_t m = Vertices[0].size();
        uint64_t ind = m;
        Vector<V> pointProj = point;
        V furthest = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < Vertices.size(); ++i) {
            const Vector<V> &w = Directions[i];
            V distance = w.dot(point - Vertices[i]) / std::pow(w.template lpNorm<2>(), 2);
            if (distance > furthest) {
                furthest = distance;
                ind = i;
            }
        }
        if (ind < m) {
            pointProj = point - (furthest * Directions[ind]);
        }
        return pointProj;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::argminUnitSimplexProjection(Vector<V> &iniPoint,
                                                                       const std::vector<Vector<V>> &Phi) {

        mopmc::optimization::auxiliary::LinearCombination<V> lincom(this->fn, Phi);

        uint64_t maxIter = 1e5;
        uint64_t k = Phi.size();
        uint64_t m = Phi[0].size();
        V gamma = static_cast<V>(0.1);
        V epsilon = static_cast<V>(1.e-5);
        V error;
        Vector<V> alphaCurrent = iniPoint, alphaNew(k), alphaTemp(k);
        Vector<V> grad(k);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            this->fn->subgradient(Phi[0]);
            grad = lincom.gradient(alphaCurrent);
            alphaTemp = alphaCurrent - gamma * grad * 0.5 * std::log(2 + it);// * 2.0 / (2 + it);
            alphaNew = projectToUnitSimplex(alphaTemp);
            error = (alphaNew - alphaCurrent).template lpNorm<1>();
            if (error < epsilon) {
                alphaCurrent = alphaNew;
                break;
            }
            alphaCurrent = alphaNew;
        }
        std::cout << "*Projected GD* (to simplex) stops at iteration " << it << " with error: " << error << "\n";
        Vector<V> result(m);
        result.setZero();
        for (uint_fast64_t i = 0; i < k; ++i) {
            result += alphaNew(i) * Phi[i];
        }
        return result;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::projectToUnitSimplex(Vector<V> &x) {
        assert(x.size() > 0);
        uint64_t m = x.size();
        //std::vector<uint64_t> ids = argsort(x);
        std::vector<uint64_t> ids = mopmc::optimization::auxiliary::Sorting<V>::argsort(x, mopmc::optimization::auxiliary::SORTING_DIRECTION::DECENT);
        V tmpsum = static_cast<V>(0.), tmax;
        bool bget = false;
        for (uint_fast64_t i = 0; i < m - 1; ++i) {
            tmpsum += x(ids[i]);
            tmax = (tmpsum - static_cast<V>(1.)) / static_cast<V>(i);
            if (tmax >= x(ids[i + 1])) {
                bget = true;
                break;
            }
        }

        if (!bget) {
            tmax = (tmpsum + x(ids[m - 1]) - static_cast<V>(1.)) / static_cast<V>(m);
        }

        Vector<V> xProj(m);
        for (uint_fast64_t j = 0; j < m; ++j) {
            xProj(ids[j]) = std::max(x(ids[j]) - tmax, static_cast<V>(0.));
        }
        return xProj;
    }

    template class ProjectedGradientDescent<double>;
}// namespace mopmc::optimization::optimizers
