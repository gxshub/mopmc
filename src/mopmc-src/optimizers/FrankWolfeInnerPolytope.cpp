//
// Created by guoxin on 24/11/23.
//

#include "FrankWolfeInnerPolytope.h"
#include "mopmc-src/auxiliary/Lincom.h"
#include "mopmc-src/auxiliary/Sorting.h"
#include "mopmc-src/auxiliary/Trigonometry.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfeInnerPolytope<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) {
        initialize(Vertices);
        const uint64_t maxIter = 1e3;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            if (checkExit(Vertices)) {
                break;
            }
            switch (this->fwOption) {
                case SIMPLEX_GD: {
                    performSimplexGradientDescent(Vertices);
                    break;
                }
                case AWAY_STEP: {
                    performForwardOrAwayStepDescent(Vertices);
                    break;
                }
                default: {
                    throw std::runtime_error("Selected FW option not implemented in this version");
                }
            }
            ++t;
        }
        std::cout << "Inner optimization, FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        point = xNew;
        return 0;
    }

    template<typename V>
    bool FrankWolfeInnerPolytope<V>::checkExit(const std::vector<Vector<V>> &Vertices) {
        const V cosTolerance = std::cos(90.0001 / 180.0 * M_PI);
        bool exit = false;
        V cosMin = 1.;
        for (int i = 0; i < size; ++i) {
            const V cos = mopmc::optimization::auxiliary::Trigonometry<V>::cosine(Vertices[i] - xCurrent, dXCurrent, 0.);
            if (cosMin > cos) {
                cosMin = cos;
            }
        }
        if (cosMin > cosTolerance) {
            std::cout << "FW loop exits due to small cosine (" << cosMin << ")\n";
            exit = true;
        }
        return exit;
    }

    template<typename V>
    void FrankWolfeInnerPolytope<V>::initialize(const std::vector<Vector<V>> &Vertices) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");
        const auto prvSize = size;
        size = Vertices.size();
        dimension = Vertices[0].size();
        xCurrent.resize(dimension);
        xNew.resize(dimension);
        xNewTmp.resize(dimension);
        alpha.conservativeResize(size);
        for (uint64_t i = prvSize; i < size; ++i) {
            alpha(i) = static_cast<V>(0.);
        }
        if (size == 1) {
            alpha(0) = static_cast<V>(1.);
            activeVertices.insert(0);
        }
        xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices, alpha);
    }


    template<typename V>
    void FrankWolfeInnerPolytope<V>::performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {
        Vector<V> dAlpha = Vector<V>::Zero(size);
        for (int64_t i = 0; i < size; ++i) {
            dAlpha(i) += dXCurrent.dot(Vertices[i]);
        }
        auto sortedIndices = mopmc::optimization::auxiliary::Sorting<V>::ascendingArgsort(dAlpha);
        Vector<V> dAlphaTmp(size);
        int64_t offsetIndex, numNullVertices = size - activeVertices.size();
        for (offsetIndex = 0; offsetIndex < size; ++offsetIndex) {
            if (!activeVertices.count(sortedIndices[offsetIndex])) {
                numNullVertices -= 1;
            }
            for (uint64_t i = 0; i < size; ++i) {
                if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                    dAlphaTmp(sortedIndices[i]) = dAlpha(sortedIndices[i]) - dAlpha(sortedIndices[offsetIndex]);
                } else {
                    dAlphaTmp(sortedIndices[i]) = static_cast<V>(0.);
                }
            }
            if (dAlphaTmp.sum() <= 0) {
                break;
            }
        }
        if (offsetIndex == 0) {
            dAlpha.setZero();
            return;
        }
        const V offset = dAlpha.sum() / (size - numNullVertices);
        for (int64_t i = 0; i < size; ++i) {
            if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                dAlpha(sortedIndices[i]) -= offset;
            } else {
                dAlpha(sortedIndices[i]) = 0.;
            }
        }
        V step = std::numeric_limits<V>::max();
        uint64_t resetIndex = size;
        for (uint64_t vertexIndex = 0; vertexIndex < size; ++vertexIndex) {
            if (dAlpha(vertexIndex) > 0.) {
                if (alpha(vertexIndex) / dAlpha(vertexIndex) < step) {
                    step = alpha(vertexIndex) / dAlpha(vertexIndex);
                    resetIndex = vertexIndex;
                }
            }
        }
        assert(resetIndex != size);
        xNewTmp = xCurrent;
        for (uint64_t i = 0; i < size; ++i) {
            xNewTmp -= (step * dAlpha(i)) * Vertices[i];
        }
        V gamma = static_cast<V>(1.0);
        if (this->fn->value(xCurrent) >= this->fn->value(xNewTmp)) {
            xNew = xNewTmp;
            activeVertices.erase(resetIndex);
        } else {
            gamma = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp, 1.0);
            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
        }
        alpha -= (gamma * step) * dAlpha;
        for (uint64_t i = 0; i < size; ++i) {
            if (dAlpha(i) < 0. && !activeVertices.count(i)) {
                activeVertices.insert(i);
            }
        }
        //rescale alpha to reduce rounding error
        if (alpha.sum() < 1.0) {
            alpha /= alpha.sum();
        }
    }

    template<typename V>
    void FrankWolfeInnerPolytope<V>::performForwardOrAwayStepDescent(const std::vector<Vector<V>> &Vertices) {
        V gamma, gammaMax, epsFwd, epsAwy;
        uint64_t fwdInd{}, awyInd{};
        Vector<V> fwdVec, awyVec;
        bool isFwd{};
        checkForwardStep(Vertices, fwdInd, fwdVec, epsFwd);
        checkAwayStep(Vertices, awyInd, awyVec, epsAwy);
        forwardOrAwayStepUpdate(fwdInd, fwdVec, awyInd, awyVec, gamma, gammaMax, isFwd);
    }

    template<typename V>
    void FrankWolfeInnerPolytope<V>::forwardOrAwayStepUpdate(uint64_t &fwdInd, Vector<V> &fwdVec,
                                                             uint64_t &awyInd, Vector<V> &awyVec,
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

        gamma = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp, gammaMax);

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
    void FrankWolfeInnerPolytope<V>::checkAwayStep(const std::vector<Vector<V>> &Vertices, uint64_t &awyInd, Vector<V> &awyVec, V &awyEps) {
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
    void FrankWolfeInnerPolytope<V>::checkForwardStep(const std::vector<Vector<V>> &Vertices, uint64_t &fwdInd, Vector<V> &fwdVec, V &fwdEps) {
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

    template class FrankWolfeInnerPolytope<double>;
}// namespace mopmc::optimization::optimizers
