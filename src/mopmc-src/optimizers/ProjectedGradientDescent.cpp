//
// Created by guoxin on 4/12/23.
//

#include "ProjectedGradientDescent.h"
#include "../auxiliary/Lincom.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

namespace mopmc::optimization::optimizers {

    /**
    * Argsort(for descending sort)
    * @param V array element type
    * @param vec input array
    * @return indices w.r.t sorted array
    */
    template<typename V>
    std::vector<size_t> argsort(const Vector<V> &vec) {
        std::vector<size_t> indices(vec.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&vec](int left, int right) -> bool {
                      // sort indices according to corresponding array element
                      return vec(left) > vec(right);
                  });

        return indices;
    }

    template<typename V>
    ProjectedGradientDescent<V>::ProjectedGradientDescent(ProjectionType type,
                                                          convex_functions::BaseConvexFunction<V> *f)
        : projectionType(type), BaseOptimizer<V>(f) {}


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
        this->point_ = argminUnitSimplexProjection(this->alpha, Vertices);
        point = this->point_;
        return 0;
    }

    template<typename V>
    int ProjectedGradientDescent<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices,
                                              const std::vector<Vector<V>> &Weights) {
        assert(this->projectionType == ProjectionType::NearestHyperplane);
        assert(!Vertices.empty() && Vertices.size() == Weights.size());
        this->point_ = argminNearestHyperplane(point, Vertices, Weights);
        point = this->point_;
        return 0;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::argminNearestHyperplane(Vector<V> &iniPoint,
                                                                   const std::vector<Vector<V>> &Phi,
                                                                   const std::vector<Vector<V>> &W) {

        uint64_t maxIter = 1000;
        uint64_t m = iniPoint.size();
        V gamma = static_cast<V>(0.1);
        V epsilon = static_cast<V>(1.e-8);
        Vector<V> xCurrent = iniPoint, xNew(m), xTemp(m);
        Vector<V> grad(m);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            grad = this->fn->subgradient(xCurrent);
            xTemp = xCurrent - gamma * grad;//it + static_cast<V>(1.))
            xNew = projectToNearestHyperplane(xTemp, Phi, W);
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                break;
            }
            xCurrent = xNew;
        }
        std::cout << "*Projected GD* stops at iteration " << it << "\n";
        return xCurrent;
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
    Vector<V> ProjectedGradientDescent<V>::projectToNearestHyperplane(Vector<V> &x,
                                                                      const std::vector<Vector<V>> &Phi,
                                                                      const std::vector<Vector<V>> &W) {

        assert(W.size() == Phi.size());
        assert(!Phi.empty());
        uint64_t m = Phi[0].size();
        V shortest = std::numeric_limits<V>::max();
        Vector<V> xProj = x;
        for (uint_fast64_t i = 0; i < Phi.size(); ++i) {
            V e = Phi[i].template lpNorm<2>();
            V distance = W[i].dot(x - Phi[i]) / std::pow(e, 2);
            if (distance > 0 && distance < shortest) {
                shortest = distance;
                xProj = x - (shortest * W[i]);
            }
        }
        return xProj;
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
