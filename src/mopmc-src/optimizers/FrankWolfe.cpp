//
// Created by guoxin on 24/11/23.
//

#include "FrankWolfe.h"
#include "../auxiliary/Lincom.h"
#include "../auxiliary/Sorting.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    V cosine(const Vector<V> &x, const Vector<V> &y, const V &c) {
        V a = x.template lpNorm<2>();
        V b = y.template lpNorm<2>();
        if (a == 0 || b == 0) {
            return c;
        } else {
            return x.dot(y) / (a * b);
        }
    }

    template<typename V>
    int FrankWolfe<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) {
        point = argmin(Vertices);
        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(const std::vector<Vector<V>> &Vertices) {

        initialize(Vertices);
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            computeForwardStepIndexAndVector(Vertices);
            computeAwayStepIndexAndVector(Vertices);
            epsFwd = static_cast<V>(-1.) * dXCurrent.dot(Vertices[fwdInd] - xCurrent);
            epsAwy = static_cast<V>(-1.) * dXCurrent.dot(xCurrent - Vertices[awyInd]);
            if (epsFwd <= tolerance) {
                std::cout << "FW loop exits due to small tolerance: " << epsFwd << "\n";
                ++t;
                break;
            }
            if (cosine(fwdVec, dXCurrent, 0.) > toleranceCosine) {
                std::cout << "FW loop exits due to small cosine: " << cosine(fwdVec, dXCurrent, 0.) << "\n";
                ++t;
                break;
            }

            switch (this->fwOption) {
                case LINOPT: {
                    PolytopeType polytopeType = PolytopeType::Vertex;
                    this->linOpt.optimizeVtx(Vertices, polytopeType, dXCurrent, xNewEx);
                    gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx);
                    xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                    break;
                }
                case AWAY_STEP: {
                    updateWithForwardOrAwayStep();
                    //std::cout<< "xNew: " << xNew << "\n";
                    break;
                }
                case BLENDED: {
                    if (epsFwd + epsAwy >= delta) {
                        updateWithForwardOrAwayStep();
                    } else {
                        int feasible = -1;
                        this->linOpt.checkPointInConvexHull(Vertices, (xCurrent - dXCurrent * delta), feasible);
                        if (feasible == 0) {
                            xNewEx = xCurrent - dXCurrent * delta;
                            gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, static_cast<V>(1.));
                            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                        } else if (feasible == 2) {
                            delta *= static_cast<V>(0.5);
                        } else {
                            printf("[Warning] ret = %i\n", feasible);
                            ++t;
                            break;
                            //throw std::runtime_error("linopt error");
                        }
                    }
                    break;
                }
                case BLENDED_STEP_OPT: {
                    this->linOpt.findMaximumFeasibleStep(Vertices, dXCurrent, xCurrent, stepSize);
                    if (stepSize > delta * scale2) {
                        xNewEx = xCurrent - dXCurrent * stepSize;
                        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, static_cast<V>(1.));
                        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                    } else {
                        delta *= static_cast<V>(0.5);
                    }
                    break;
                }
                case SIMPLEX_GD: {
                    updateWithSimplexGradientDescent(Vertices);
                    //std::cout<< "xNew: " << xNew << "\n";
                    break;
                }
            }
            ++t;
        }
        std::cout << "Frank-Wolfe stops at iteration: " << t << "\n";
        return xNew;
    }

    template<typename V>
    void FrankWolfe<V>::initialize(const std::vector<Vector<V>> &Vertices) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        size = Vertices.size();
        dimension = Vertices[0].size();
        xCurrent.resize(dimension);
        xNew.resize(dimension);
        xNewEx.resize(dimension);

        alpha.conservativeResize(size);
        alpha(size - 1) = static_cast<V>(0.);

        if (size == 1) {
            alpha(0) = static_cast<V>(1.);
            activeSet.insert(0);
        }

        xNew.setZero();
        for (uint_fast64_t i = 0; i < size; ++i) {
            assert(xNew.size() == Vertices[i].size());
            xNew += this->alpha(i) * Vertices[i];
        }

        //estimate initial gap
        delta = std::numeric_limits<V>::min();
        for (uint_fast64_t i = 0; i < size; ++i) {
            const V c = (this->fn->gradient(xNew)).dot(xNew - Vertices[i]) / scale1;
            if (c > delta) {
                delta = c;
            }
        }
    }

    template<typename V>
    void FrankWolfe<V>::updateWithSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {
        Vector<V> dAlpha = Vector<V>::Zero(size);
        for (int64_t i = 0; i < size; ++i) {
            dAlpha(i) += dXCurrent.dot(Vertices[i]);
        }
        auto valueIndices = mopmc::optimization::auxiliary::Sorting<V>::argsort(dAlpha, mopmc::optimization::auxiliary::SORTING_DIRECTION::ASCENT);
        Eigen::ArrayXd dAlphaAry = dAlpha.array();
        Eigen::ArrayXd dAlphaAryTmp = dAlphaAry;
        int64_t pivot, nNullVertices = 0;
        for (pivot = 0; pivot < size; ++pivot) {
            dAlphaAryTmp = dAlpha.array() - dAlpha(valueIndices[pivot]);
            for (int64_t j = pivot; j < size; ++j) {
                if (!activeSet.count(valueIndices[j])) {
                    dAlphaAryTmp(valueIndices[j]) = 0.;
                    nNullVertices += 1;
                }
            }
            if (dAlphaAryTmp.sum() <= 0) {
                break;
            }
        }
        if (pivot == 0 || pivot == size) {
            dAlpha.setZero();
            return;
        }
        dAlphaAryTmp -= (dAlphaAryTmp.sum() / (size - nNullVertices));
        for (int64_t j = pivot; j < size; ++j) {
            dAlphaAryTmp = static_cast<V>(0.);
        }
        dAlpha = dAlphaAryTmp.matrix();
        V lambda = std::numeric_limits<V>::max();
        uint64_t ind = size;
        for (uint64_t i = 0; i < size; ++i) {
            if (dAlpha(i) > 0.) {
                if (alpha(i) / dAlpha(i) < lambda) {
                    lambda = alpha(i) / dAlpha(i);
                    ind = i;
                }
            }
        }
        if (ind == size) {
            std::cout << "alpha: " << alpha << "\n";
            std::cout << "dAlpha: " << dAlpha << "\n";
        }
        assert(ind != size);
        xNewEx = xCurrent;
        for (uint64_t i = 0; i < size; ++i) {
            xNewEx -= (lambda * dAlpha(i)) * Vertices[i];
        }
        if (this->fn->value(xNewEx) <= this->fn->value(xCurrent)) {
            xNew = xNewEx;
            gamma = 1.0;
        } else {
            gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, 1.0);
            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
        }
        alpha -= lambda * dAlpha;
        for (uint64_t i = 0; i < size; ++i) {
            if (dAlpha(i) < 0.) {
                activeSet.insert(i);
            }
        }
        if (gamma == 1.0) {
            activeSet.erase(ind);
        }
        assert(alpha.sum() > 0.);
    }

    template<typename V>
    void FrankWolfe<V>::updateWithForwardOrAwayStep() {
        if (static_cast<V>(-1.) * dXCurrent.dot(fwdVec - awyVec) >= 0.) {
            isFwd = true;
            xNewEx = xCurrent + fwdVec;
            gammaMax = static_cast<V>(1.);
        } else {
            isFwd = false;
            xNewEx = xCurrent + awyVec;
            gammaMax = this->alpha(awyInd) / (static_cast<V>(1.) - this->alpha(awyInd));
        }

        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, gammaMax);

        if (isFwd) {
            if (gamma == gammaMax) {
                this->activeSet.clear();
                this->activeSet.insert(fwdInd);
            } else {
                this->activeSet.insert(fwdInd);
            }

            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != fwdInd) {
                    this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                }
            }
            this->alpha(fwdInd) = (static_cast<V>(1.) - gamma) * this->alpha(fwdInd) + gamma;
        } else {
            if (gamma == gammaMax) {
                this->activeSet.erase(awyInd);
            }
            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != awyInd) {
                    this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                }
            }
            this->alpha(awyInd) = (static_cast<V>(1.) + gamma) * this->alpha(awyInd) - gamma;
        }
        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
    }

    template<typename V>
    void FrankWolfe<V>::computeAwayStepIndexAndVector(const std::vector<Vector<V>> &Vertices) {
        awyInd = 0;
        V inc = std::numeric_limits<V>::min();
        for (auto j: this->activeSet) {
            //assert(Vertices[j].size() == dXCurrent.size());
            if (Vertices[j].dot(dXCurrent) > inc) {
                inc = Vertices[j].dot(dXCurrent);
                awyInd = j;
            }
        }
        awyVec = xCurrent - Vertices[awyInd];
    }

    template<typename V>
    void FrankWolfe<V>::computeForwardStepIndexAndVector(const std::vector<Vector<V>> &Vertices) {
        fwdInd = 0;
        V dec = std::numeric_limits<V>::max();
        for (uint_fast64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(dXCurrent) < dec) {
                dec = Vertices[i].dot(dXCurrent);
                fwdInd = i;
            }
        }
        fwdVec = (Vertices[fwdInd] - xCurrent);
    }

    template<typename V>
    FrankWolfe<V>::FrankWolfe(FWOption option, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
        : fwOption(option), BaseOptimizer<V>(f) {
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
    }

    template class FrankWolfe<double>;
}// namespace mopmc::optimization::optimizers
