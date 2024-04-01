//
// Created by guoxin on 30/03/24.
//

#include "MinimumNormPoint.h"
#include "SeparationHyperplaneOptimizer.h"
#include "mopmc-src/auxiliary/Lincom.h"
#include "mopmc-src/auxiliary/Sorting.h"
#include "mopmc-src/auxiliary/Trigonometry.h"
#include "mopmc-src/convex-functions/MSE.h"
#include "lp_lib.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int MinimumNormPoint<V>::minimize(Vector<V> &sepDirection,
                                      Vector<V> &optimum, const std::vector<Vector<V>> &Vertices, const Vector<V> &pivot) {
        assert(pivot.size() == optimum.size());
        this->fn = new mopmc::optimization::convex_functions::MSE<V>(pivot, pivot.size());
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(this->fn);
        bool hasGotMaxMarginSepHP = false;
        SeparationHyperplaneOptimizer<V> separationHyperplaneOptimizer;
        initialize(Vertices);
        //std::cout << "[Minimum norm point optimization] Vertices size: " << size << "\n";
        const uint64_t maxIter = 100;// 1e3;
        uint64_t t = 0;
        if (Vertices.size() == 1) {
            optimum = Vertices[0];
            sepDirection = (pivot - Vertices[0])/(pivot - Vertices[0]).template lpNorm<1>();
            return 0;
        }
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            //std::cout << "[before checkSeparation] GOT HERE\n";
            if (!hasGotMaxMarginSepHP && checkSeparation(Vertices, pivot - xNew, pivot)) {
                hasGotMaxMarginSepHP = true;
                Vector<V> sign(dimension);
                for (uint64_t i; i < sign.size(); ++i) {
                    if ((pivot - xNew)(i) >= 0) {
                        sign(i) = static_cast<V>(1.);
                    } else {
                        sign(i) = static_cast<V>(-1.);
                    }
                }
                V margin;
                separationHyperplaneOptimizer.findMaximumSeparatingDirection(Vertices, pivot, sign, sepDirection, margin);
                std::cout << "[Minimum norm point optimization] computed maximum margin separation hyperplane\n";
            }
            /*
            if (checkExit(Vertices)) {
                break;
            }
             */
            performSimplexGradientDescent(Vertices);
            optimum = xNew;
            ++t;
        }
        std::cout << "[Minimum norm point optimization] FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        return 0;
    }

    template<typename V>
    int MinimumNormPoint<V>::minimize(Vector<V> &optimum,
                                      const std::vector<Vector<V>> &Vertices,
                                      const Vector<V> &pivot) {
        assert(pivot.size() == optimum.size());
        //mopmc::optimization::convex_functions::MSE<V> f(pivot, pivot.size());
        this->fn = new mopmc::optimization::convex_functions::MSE<V>(pivot, pivot.size());
        //this->fn = &f;
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(this->fn);
        SeparationHyperplaneOptimizer<V> separationHyperplaneOptimizer;
        initialize(Vertices);
        //std::cout << "[Minimum norm point optimization] Vertices size: " << size << "\n";
        const uint64_t maxIter = 20;// 1e3;
        uint64_t t = 0;
        if (Vertices.size() == 1) {
            optimum = Vertices[0];
            return 0;
        }
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            //std::cout << "[before checkSeparation] GOT HERE\n";
            if (checkSeparation(Vertices, pivot - xNew, pivot)) {
                Vector<V> sign(dimension);
                for (uint64_t i; i < sign.size(); ++i) {
                    if ((pivot - xNew)(i) >= 0) {
                        sign(i) = static_cast<V>(1.);
                    } else {
                        sign(i) = static_cast<V>(-1.);
                    }
                }
                Vector<V> dir(dimension);
                V margin, margin_crt, margin_prv;
                separationHyperplaneOptimizer.findMaximumSeparatingDirection(Vertices, pivot, sign, dir, margin);
                //std::cout << "[before findNearestPointByDirection] GOT HERE\n";
                //findNearestPointByDirection(Vertices, pivot - xNew, pivot, this->alpha);
                //std::cout << "[after findNearestPointByDirection] GOT HERE\n";
                xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices, alpha);
                optimum = xNew;
                //std::cout << "[Minimum norm point optimization] FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
                return 0;
            }
            /*
            if (checkExit(Vertices)) {
                break;
            }
             */
            performSimplexGradientDescent(Vertices);
            ++t;
        }
        std::cout << "[Minimum norm point optimization] no optimum found\n";
        return 1;
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
        std::cout << "[Minimum norm point optimization] cosMin: " << cosMin <<"\n";
        if (cosMin > cosTolerance) {
            std::cout << "[Minimum norm point optimization] exits due to small angle (cosine: " << cosMin <<  ")\n";
            exit = true;
        }
        return exit;
    }

    template<typename V>
    bool MinimumNormPoint<V>::checkSeparation(const std::vector<Vector<V>> &Vertices, const Vector<V> &direction, const Vector<V> &point) {
        bool exit = true;
        for (uint64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(direction) >= point.dot(direction)) {
                exit = false;
                break;
            }
        }
        return exit;
    }

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
        //if (xCurrent.size() != dimension) {
            xCurrent.resize(dimension);
            xNew.resize(dimension);
            xNewTmp.resize(dimension);
        //}
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
    void MinimumNormPoint<V>::performSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {

        Vector<V> dAlpha(size), dAlphaTmp(size);
        for (int64_t i = 0; i < size; ++i) {
            //projected gradient on vertex i
            dAlpha(i) = dXCurrent.dot(Vertices[i]);
        }
        auto sortedIndices = mopmc::optimization::auxiliary::Sorting<V>::ascendingArgsort(dAlpha);
        int64_t offsetIndex, numNullVertices = size - activeVertices.size();
        for (offsetIndex = 0; offsetIndex < size; ++offsetIndex) {
            for (uint64_t i = 0; i < size; ++i) {
                if (i > offsetIndex && !activeVertices.count(sortedIndices[i])) {
                    dAlphaTmp(sortedIndices[i]) = static_cast<V>(0.);
                } else {
                    dAlphaTmp(sortedIndices[i]) -= dAlpha(sortedIndices[offsetIndex]);
                }
                /*
                if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                    dAlphaTmp(sortedIndices[i]) = dAlpha(sortedIndices[i]) - dAlpha(sortedIndices[offsetIndex]);
                } else {
                    dAlphaTmp(sortedIndices[i]) = static_cast<V>(0.);
                }*/
            }
            if (dAlphaTmp.sum() <= 0) {
                break;
            } else if (!activeVertices.count(sortedIndices[offsetIndex])) {
                numNullVertices -= 1;
            }
        }
        assert(size > numNullVertices);
        const V offset1 = dAlphaTmp.sum() / (size - numNullVertices);
        //std::cout << "offset1: " << offset1 <<"\n";
        for (int64_t i = 0; i < size; ++i) {
            if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                dAlphaTmp(sortedIndices[i]) -= offset1;
            }
        }
        //std::cout << "dAlphaTmp: " << dAlphaTmp <<"\n";
        if (offsetIndex == 0) {
            dAlpha.setZero();
            return;
        }
        if (dAlphaTmp.isZero()) {
            return;
        }
        dAlpha = dAlphaTmp;
        /*
        const V offset = dAlpha.sum() / (size - numNullVertices);
        for (int64_t i = 0; i < size; ++i) {
            if (i < offsetIndex || activeVertices.count(sortedIndices[i])) {
                dAlpha(sortedIndices[i]) -= offset;
            } else {
                dAlpha(sortedIndices[i]) = 0.;
            }
        }*/
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
        //std::cout << "dAlpha: " << dAlpha <<"\n";
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
            std::cout << "[In simple gradient desc, else cond] GOT HERE\n";
            gamma = this->lineSearcher.findOptimalRelativeDistance(xCurrent, xNewTmp, 1.0);
            //std::cout << "[In simple gradient desc, else cond] GOT HERE\n";
            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewTmp;
        }
        //std::cout << "[In simple gradient desc] GOT HERE\n";
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
        xNew = mopmc::optimization::auxiliary::LinearCombination<V>::combine(Vertices,alpha);
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
