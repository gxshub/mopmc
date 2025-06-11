//
// Created by guoxin on 30/03/24.
//

#include "MinimumNormPoint.h"
#include "MaximumMarginSeparationHyperplane.h"
#include "lp_lib.h"
#include "mopmc-src/auxiliary/Lincom.h"
#include "mopmc-src/auxiliary/Sorting.h"
#include "mopmc-src/auxiliary/Trigonometry.h"
#include "mopmc-src/convex-functions/MSE.h"
#include "mopmc-src/Printer.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int MinimumNormPoint<V>::optimizeSeparationDirection(Vector<V> &sepDirection,
                                      Vector<V> &optimum,
                                      V &margin,
                                      const std::vector<Vector<V>> &Vertices,
                                      const Vector<V> &pivot) {
        assert(pivot.size() == optimum.size());
        this->fn = new mopmc::optimization::convex_functions::MSE<V>(pivot, pivot.size());
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(this->fn);
        bool hasFoundMaxMarginSepHp(false);
        MaximumMarginSeparationHyperplane<V> separationHyperplaneOptimizer;
        initialize(Vertices);
        const uint64_t maxIter0 = 1e3;
        const uint64_t maxIter1 = 5e2;
        if (Vertices.size() == 1) {
            optimum = Vertices[0];
            sepDirection = (pivot - Vertices[0])/((pivot - Vertices[0]).template lpNorm<1>());
            std::cout << "[Minimum norm point optimization] exits for one vertex\n";
            return EXIT_SUCCESS;
        }
        uint64_t t = 0;
        while (t < maxIter0) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            performSimplexGradientDescent(Vertices);
            if (this->fn->value(xCurrent) - this->fn->value(xNew) < 1e-12) {
                break;
            }
            ++t;
        }
        xCurrent = xNew;
        uint64_t t1 = 0;
        while (t1 < maxIter1){
            auto pair = getMaximumMarginPoint(Vertices, pivot - xCurrent, pivot);
            if (pair.second > 1e-12){
                const int64_t idx = pair.first;
                const Vector<V> & xTmp= Vertices[idx];
                V gamma = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xTmp, 1.0);
                xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xTmp;
                this->alpha = (static_cast<V>(1.) - gamma) * this->alpha;
                this->alpha(idx) += gamma;
            } else{
                hasFoundMaxMarginSepHp = true;
                break;
            }
            xCurrent = xNew;
            ++t1;
        }
        sepDirection = pivot - xNew;
        optimum = xNew;
        assert(alpha.size() == Vertices.size());
        if (hasFoundMaxMarginSepHp) {
            std::cout << "[Minimum norm point optimization] max margin separation hyperplane computed, terminates at iteration: " << t + t1<<  " (distance: " << this->fn->value(xNew) << ")\n";
            return EXIT_SUCCESS;
        } else {
            std::cout << "[Minimum norm point optimization] no separation hyperplane found after " << maxIter0 + maxIter1 << " iterations\n";
            return EXIT_FAILURE;
        }
    }

    template<typename V>
    bool MinimumNormPoint<V>::checkSeparation(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point) {
        bool exit = true;
        const V delta = 1e-6;
        for (uint64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(direction) >= point.dot(direction) - delta) {
                exit = false;
                break;
            }
        }
        return exit;
    }

    template<typename V>
    std::pair<int64_t, V> MinimumNormPoint<V>::getMaximumMarginPoint(const std::vector<Vector<V>> &Vertices,
                                                                     const Vector<V> &direction,
                                                                     const Vector<V> &point){
        const V delta = 1e-12;
        int64_t idx = -1;
        V margin = 0;
        for (int64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(direction) - point.dot(direction) > margin) {

                idx = i;
                margin = Vertices[i].dot(direction) - point.dot(direction);
            }
        }
        return {idx, margin};
    };


    template<typename V>
    V MinimumNormPoint<V>::getSeparationMargin(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point) {
        V margin = std::numeric_limits<V>::min();
        for (uint64_t i = 0; i < Vertices.size(); ++i) {
            V tmp = (point - Vertices[i]).dot(direction);
            if (margin < tmp) {
                margin = tmp;
            }
        }
        return margin;
    }

    template<typename V>
    void MinimumNormPoint<V>::initialize(const std::vector<Vector<V>> &Vertices) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");
        //initial size is 0
        const auto prvSize = size;
        //update size
        size = Vertices.size();
        dimension = Vertices[0].size();
        if (xCurrent.size() != dimension) {
            xCurrent.resize(dimension);
            xNew.resize(dimension);
            xNewTmp.resize(dimension);
        }
        alpha.conservativeResize(size);
        for (uint64_t i = prvSize; i < size; ++i) {
            alpha(i) = static_cast<V>(0.);
        }
        if (prvSize == 0) {
            alpha(0) = static_cast<V>(1.);
            activeVertices.insert(0);
        }
        xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices, alpha);
    }

    template<typename V>
    void MinimumNormPoint<V>::performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {
        const V epsilon = 1e-12;
        Vector<V> dAlpha = Vector<V>::Zero(size);
        for (int64_t i = 0; i < size; ++i) {
            dAlpha(i) += dXCurrent.dot(Vertices[i]);
        }
        auto sortedIndices = mopmc::optimization::auxiliary::Sorting<V>::ascendingArgsort(dAlpha);
        Vector<V> dAlphaTmp(size);
        int64_t offsetIndex, numNullVertices = size - activeVertices.size();
        for (offsetIndex = 0; offsetIndex < size; ++offsetIndex) {
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
            if (!activeVertices.count(sortedIndices[offsetIndex])) {
                numNullVertices -= 1;
            }
        }
        if (offsetIndex == 0) {
            dAlpha.setZero();
            return;
        }
        const V offset = dAlphaTmp.sum() / (size - numNullVertices);
        for (int64_t i = 0; i < size; ++i) {
            if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                dAlphaTmp(sortedIndices[i]) -= offset;
            }
        }
        V dAlphaTmpNorm = dAlphaTmp.template lpNorm<1>();
        if (dAlphaTmpNorm < epsilon) {
            return;
        } else {
            dAlpha = dAlphaTmp / dAlphaTmp.template lpNorm<1>();
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
        if (resetIndex == size) {
            mopmc::Printer<V>::printVector(" dAlpha ", dAlpha);
            mopmc::Printer<V>::printVector(" dXCurrent ", dXCurrent);
            mopmc::Printer<V>::printVector(" this->fn->parameters ", this->fn->parameters);
            std::cout << " this->fn->dimension : " << this->fn->dimension <<"\n";
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
    int MinimumNormPoint<V>::findNearestPointByDirection(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point,
                                                         Vector<V> &weights) {
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Vertices.empty());
        n_cols = Vertices.size() + 1;// number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1;// couldn't construct a new model
        if (ret == 0) {
            //[important!] set the unbounded variables.
            // The default bounds are >=0 in lp solve.
            //set_unbounded(lp, n_cols);
            // create space large enough for one row
            col_no = (int *) malloc(n_cols * sizeof(*col_no));
            row = (V *) malloc(n_cols * sizeof(*row));
            if ((col_no == NULL) || (row == NULL))
                ret = 2;
        }
        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[j] = j + 1;
                row[j] = static_cast<V>(1.);
            }
            if (!add_constraintex(lp, n_cols - 1, row, col_no, EQ, static_cast<V>(1.)))
                ret = 3;
            //std::cout << "[in findNearestPointByDirection, add constraint] GOT HERE\n";
            for (int i = 0; i < point.size(); ++i) {
                for (int j = 0; j < n_cols - 1; ++j) {
                    col_no[j] = j + 1;
                    row[j] = Vertices[j](i);
                    std::cout << "[in findNearestPointByDirection, add constraint] GOT HERE\n, j: " << j<<"\n";
                }
                col_no[n_cols] = n_cols + 1;
                row[n_cols] = direction(i);
                //assert(point.size() == direction.size());
                //std::cout << "[in findNearestPointByDirection] size of point: " << point.size()<<"\n";
                std::cout << "[in findNearestPointByDirection, add constraint (0)] GOT HERE\n, i: " << i<<"\n";
                if (!add_constraintex(lp, n_cols, row, col_no, EQ, point(i)))
                    ret = 3;
                std::cout << "[in findNearestPointByDirection, add constraint (1)] GOT HERE\n, i: " << i<<"\n";
            }
        }
        std::cout << "[in findNearestPointByDirection, after adding constraints] GOT HERE\n";
        if (ret == 0) {
            set_add_rowmode(lp, FALSE);// rowmode should be turned off again when done building the model
            col_no[0] = n_cols;
            row[0] = static_cast<V>(1.); /* set the objective in lpsolve */
            if (!set_obj_fnex(lp, 1, row, col_no))
                ret = 4;
        }
        if (ret == 0) {
            // set the object weightVector to maximize
            set_minim(lp);
            //write_LP(lp, stdout);
            // write_lp(lp, "model.lp");
            set_verbose(lp, IMPORTANT);
            ret = solve(lp);
            //std::cout<< "** Optimal solution? Ret: " << ret << "\n";
            if (ret == OPTIMAL)
                ret = 0;
            else
                ret = 5;
        }
        if (ret == 0) {
            get_variables(lp, row);
            // we are done now
        } else {
            std::cout << "error in optimization (Ret = " << ret << ")\n";
        }
        weights = VectorMap<V>(row, n_cols - 1);
        return 0;
    }

    template class MinimumNormPoint<double>;
}// namespace mopmc::optimization::optimizers
