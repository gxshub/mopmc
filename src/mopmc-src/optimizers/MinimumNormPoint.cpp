//
// Created by guoxin on 30/03/24.
//

#include "MinimumNormPoint.h"
#include "mopmc-src/auxiliary/Lincom.h"
#include "mopmc-src/auxiliary/Sorting.h"
#include "mopmc-src/auxiliary/Trigonometry.h"
#include "mopmc-src/convex-functions/MSE.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int MinimumNormPoint<V>::minimize(Vector<V> &optimum,
                                      const std::vector<Vector<V>> &Vertices,
                                      const Vector<V> &pivot) {
        assert(pivot.size() == optimum.size());
        this->fn = new mopmc::optimization::convex_functions::MSE<V>(pivot, pivot.size());
        initialize(Vertices);
        const uint64_t maxIter = 1e3;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            if (checkExit(Vertices)) {
                break;
            }
            performSimplexGradientDescent(Vertices);
            ++t;
        }
        std::cout << "Minimum norm point optimization, FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        optimum = xNew;
        return 0;
    }

    template<typename V>
    bool MinimumNormPoint<V>::checkExit(const std::vector<Vector<V>> &Vertices) {
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
    void MinimumNormPoint<V>::initialize(const std::vector<Vector<V>> &Vertices) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");
        const auto sizePrv = size;
        size = Vertices.size();
        dimension = Vertices[0].size();
        xCurrent.resize(dimension);
        xNew.resize(dimension);
        xNewTmp.resize(dimension);
        alpha.conservativeResize(size);
        for (uint64_t i = sizePrv; i < size; ++i) {
            alpha(i) = static_cast<V>(0.);
        }
        if (sizePrv == 0) {
            alpha(0) = static_cast<V>(1.);
            activeVertices.insert(0);
        }
        xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices, alpha);
    }

    template<typename V>
    void MinimumNormPoint<V>::performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {
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

    template class MinimumNormPoint<double>;
}// namespace mopmc::optimization::optimizers
