//
// Created by guoxin on 24/11/23.
//

#include "FrankWolfe.h"
#include "../auxiliary/Lincom.h"
#include "../auxiliary/Sorting.h"
#include "../auxiliary/Trigonometry.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfe<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) {
        point = argmin(Vertices);
        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(const std::vector<Vector<V>> &Vertices) {

        const uint64_t maxIter = 1e3;
        const V tolerance{1.e-8}, toleranceCosine = std::cos(90.01 / 180. * M_PI);
        const V scale1{0.5}, scale2{0.5};

        V gamma, gammaMax, epsFwd, epsAwy, stepSize, delta;
        uint64_t fwdInd{}, awyInd{};
        Vector<V> fwdVec, awyVec;
        bool isFwd{};

        initialize(Vertices, delta, scale2);
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            checkForwardStep(Vertices, fwdInd, fwdVec, epsFwd);
            checkAwayStep(Vertices, awyInd, awyVec, epsAwy);
            if (epsFwd <= tolerance) {
                std::cout << "FW loop exits due to small tolerance: " << epsFwd << "\n";
                ++t;
                break;
            }

            if (mopmc::optimization::auxiliary::Trigonometry<V>::cosine(fwdVec, dXCurrent, 0.) > toleranceCosine) {
                std::cout << "FW loop exits due to small cosine: " << mopmc::optimization::auxiliary::Trigonometry<V>::cosine(fwdVec, dXCurrent, 0.) << "\n";
                ++t;
                break;
            }

            switch (this->fwOption) {
                case LINOPT: {
                    PolytopeType polytopeType = PolytopeType::Vertex;
                    this->linOpt.optimizeVtx(Vertices, polytopeType, dXCurrent, xNewTmp);
                    gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewTmp);
                    xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
                    break;
                }
                case AWAY_STEP: {
                    forwardOrAwayStepUpdate(fwdInd, fwdVec, epsFwd, awyInd, awyVec, epsAwy, gamma, gammaMax, isFwd);
                    break;
                }
                case BLENDED: {
                    if (epsFwd + epsAwy >= delta) {
                        forwardOrAwayStepUpdate(fwdInd, fwdVec, epsFwd, awyInd, awyVec, epsAwy, gamma, gammaMax, isFwd);
                    } else {
                        int feasible = -1;
                        this->linOpt.checkPointInConvexHull(Vertices, (xCurrent - dXCurrent * delta), feasible);
                        if (feasible == 0) {
                            xNewTmp = xCurrent - dXCurrent * delta;
                            gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewTmp, static_cast<V>(1.));
                            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
                        } else if (feasible == 2) {
                            delta *= static_cast<V>(0.5);
                        } else {
                            printf("[Warning] ret = %i\n", feasible);
                            ++t;
                            return xNew;
                            //throw std::runtime_error("linopt error");
                        }
                    }
                    break;
                }
                case BLENDED_STEP_OPT: {
                    this->linOpt.findMaximumFeasibleStep(Vertices, dXCurrent, xCurrent, stepSize);
                    if (stepSize > delta * scale2) {
                        xNewTmp = xCurrent - dXCurrent * stepSize;
                        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewTmp, static_cast<V>(1.));
                        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
                    } else {
                        delta *= static_cast<V>(0.5);
                    }
                    break;
                }
                case SIMPLEX_GD: {
                    simplexGradientDecentUpdate(Vertices);
                    break;
                }
            }
            std::cout << "f(xNew): " << this->fn->value(xNew) <<"\n";
            ++t;
        }
        std::cout << "Frank-Wolfe stops at iteration: " << t << "\n";
        return xNew;
    }

    template<typename V>
    void FrankWolfe<V>::initialize(const std::vector<Vector<V>> &Vertices, V &delta, const V &scale) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");
        size = Vertices.size();
        dimension = Vertices[0].size();
        xCurrent.resize(dimension);
        xNew.resize(dimension);
        xNewTmp.resize(dimension);
        alpha.conservativeResize(size);
        alpha(size - 1) = static_cast<V>(0.);
        if (size == 1) {
            alpha(0) = static_cast<V>(1.);
            activeVertices.insert(0);
        }
        xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices, alpha);
        delta = std::numeric_limits<V>::min();
        for (uint_fast64_t i = 0; i < size; ++i) {
            const V c = (this->fn->gradient(xNew)).dot(xNew - Vertices[i]) / scale;
            if (c > delta) {
                delta = c;
            }
        }
    }

    template<typename V>
    void FrankWolfe<V>::simplexGradientDecentUpdate(const std::vector<Vector<V>> &Vertices) {
        Vector<V> dAlpha = Vector<V>::Zero(size);
        for (int64_t i = 0; i < size; ++i) {
            dAlpha(i) += dXCurrent.dot(Vertices[i]);
        }
        auto valueIndices = mopmc::optimization::auxiliary::Sorting<V>::argsort(dAlpha, mopmc::optimization::auxiliary::SORTING_DIRECTION::ASCENT);
        Eigen::ArrayXd dAlphaAry = dAlpha.array();
        Eigen::ArrayXd dAlphaAryTmp = dAlphaAry;
        int64_t pivot, nNullVertices;
        for (pivot = 0; pivot < size; ++pivot) {
            dAlphaAryTmp = dAlpha.array() - dAlpha(valueIndices[pivot]);
            nNullVertices = 0;
            for (int64_t j = pivot; j < size; ++j) {
                if (!activeVertices.count(valueIndices[j])) {
                    dAlphaAryTmp(valueIndices[j]) = 0.;
                    nNullVertices += 1;
                }
            }
            if (dAlphaAryTmp.sum() <= 0) {
                break;
            }
        }
        if (pivot == 0) {
            dAlpha.setZero();
            return;
        }
        const V c = dAlphaAryTmp.sum() / (size - nNullVertices);
        for (int64_t i = 0; i < size; ++i) {
            if (i >= pivot && !activeVertices.count(valueIndices[i])) {
                dAlphaAryTmp(valueIndices[i]) = 0.;
            } else {
                dAlphaAryTmp(valueIndices[i]) -= c;
            }
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
        assert(ind != size);
        xNewTmp = xCurrent;
        for (uint64_t i = 0; i < size; ++i) {
            xNewTmp -= (lambda * dAlpha(i)) * Vertices[i];
        }
        V chi = this->fn->value(xCurrent) - this->fn->value(xNewTmp);
        V gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewTmp, 1.0);
        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
        if (this->fn->value(xCurrent) - this->fn->value(xNew) < chi / 0.75) {
            xNew = xNewTmp;
            gamma = 1.0;
        }
        alpha -= (gamma * lambda) * dAlpha;
        for (uint64_t i = 0; i < size; ++i) {
            if (dAlpha(i) < 0.) {
                activeVertices.insert(i);
            }
        }
        if (alpha.sum() <= 0.) {std::cout << "alpha before assert(alpha.sum() > 0.): " <<
        alpha <<"\n";}
        assert(alpha.sum() > 0.);
        if (gamma == 1.0) {
            activeVertices.erase(ind);
            alpha(ind) = 0.;
            alpha /= alpha.sum();
        }
    }


    template<typename V>
    void FrankWolfe<V>::forwardOrAwayStepUpdate(uint64_t &fwdInd, Vector<V> &fwdVec, V &fwdEps,
                                                uint64_t &awyInd, Vector<V> &awyVec, V &awyEps,
                                                V &gamma, V &gammaMax, bool &isFwd) {
        if (static_cast<V>(-1.) * dXCurrent.dot(fwdVec - awyVec) >= 0.) {
            isFwd = true;
            xNewTmp = xCurrent + fwdVec;
            gammaMax = static_cast<V>(1.);
        } else {
            isFwd = false;
            xNewTmp = xCurrent + awyVec;
            gammaMax = this->alpha(awyInd) / (static_cast<V>(1.) - this->alpha(awyInd));
        }

        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewTmp, gammaMax);

        if (isFwd) {
            if (gamma == gammaMax) {
                this->activeVertices.clear();
                this->activeVertices.insert(fwdInd);
            } else {
                this->activeVertices.insert(fwdInd);
            }

            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != fwdInd) {
                    this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                }
            }
            this->alpha(fwdInd) = (static_cast<V>(1.) - gamma) * this->alpha(fwdInd) + gamma;
        } else {
            if (gamma == gammaMax) {
                this->activeVertices.erase(awyInd);
            }
            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != awyInd) {
                    this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                }
            }
            this->alpha(awyInd) = (static_cast<V>(1.) + gamma) * this->alpha(awyInd) - gamma;
        }
        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
    }

    template<typename V>
    void FrankWolfe<V>::checkAwayStep(const std::vector<Vector<V>> &Vertices, uint64_t &awyInd, Vector<V> &awyVec, V &awyEps) {
        awyInd = 0;
        V inc = std::numeric_limits<V>::min();
        for (auto j: this->activeVertices) {
            if (Vertices[j].dot(dXCurrent) > inc) {
                inc = Vertices[j].dot(dXCurrent);
                awyInd = j;
            }
        }
        awyVec = xCurrent - Vertices[awyInd];
        awyEps = static_cast<V>(-1.) * dXCurrent.dot(xCurrent - Vertices[awyInd]);
    }

    template<typename V>
    void FrankWolfe<V>::checkForwardStep(const std::vector<Vector<V>> &Vertices, uint64_t &fwdInd, Vector<V> &fwdVec, V &fwdEps) {
        fwdInd = 0;
        V dec = std::numeric_limits<V>::max();
        for (uint_fast64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(dXCurrent) < dec) {
                dec = Vertices[i].dot(dXCurrent);
                fwdInd = i;
            }
        }
        fwdVec = (Vertices[fwdInd] - xCurrent);
        fwdEps = static_cast<V>(-1.) * dXCurrent.dot(Vertices[fwdInd] - xCurrent);
    }


    template<typename V>
    FrankWolfe<V>::FrankWolfe(FWOption option, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
        : fwOption(option), BaseOptimizer<V>(f) {
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
    }

    template class FrankWolfe<double>;
}// namespace mopmc::optimization::optimizers
