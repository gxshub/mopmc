//
// Created by guoxin on 26/01/24.
//

#include "FrankWolfeOuterOptimizer.h"
#include "lp_lib.h"
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int FrankWolfeOuterOptimizer<V>::minimize(Vector<V> &point,
                                             const std::vector<Vector<V>> &Vertices,
                                             const std::vector<Vector<V>> &Directions) {

        dimension = point.size();
        size = Vertices.size();
        xNew = point;
        std::set<uint64_t> exteriorHSIndices, interiorHSIndices;
        Vector<V> descentDirection(dimension);
        const uint64_t maxIter = 20;
        const V tol = 1e-12;
        bool exit = false;
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            Vector<V> slope = -1 * (this->fn->subgradient(xCurrent));
            for (uint64_t i = 0; i < size; ++i) {
                if (Directions[i].dot(Vertices[i] - xCurrent) < 1e-30) {
                    exteriorHSIndices.insert(i);
                } else {
                    interiorHSIndices.insert(i);
                }
            }
            if (exteriorHSIndices.empty()) {
                descentDirection = slope;
            } else if (exteriorHSIndices.size() == 1) {
                auto elem = exteriorHSIndices.begin();
                const Vector<V> &w = Directions[*elem];
                descentDirection = slope - (slope.dot(w)) * w;
            } else {
                findOptimalProjectedDescentDirection(Directions, exteriorHSIndices, slope,
                                                                  descentDirection);
                for (auto i: exteriorHSIndices) {
                    if (Directions[i].dot(descentDirection) > 0) {
                        exit = true;
                    }
                }
            }
            if (exit) { break; }
            V lambda = static_cast<V>(1000);
            for (auto i: interiorHSIndices) {
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
        std::cout << "Outer optimization, FW stops at iteration " << t << " (distance " << this->fn->value(xNew) << ")\n";
        //assert(this->fn->value(xNew) <= this->fn->value(point));
        point = xNew;
        return 0;
    }

    template<typename V>
    int FrankWolfeOuterOptimizer<V>::findOptimalProjectedDescentDirection(const std::vector<Vector<V>> &Directions,
                                                        const std::set<uint64_t> &exteriorIndices,
                                                        const Vector<V> &slope,
                                                        Vector<V> &descentDirection){
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!exteriorIndices.empty());
        n_cols = (int) slope.size(); // number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1; // couldn't construct a new model
        if (ret == 0) {
            // create space large enough for one row
            col_no = (int *) malloc(n_cols * sizeof(*col_no));
            row = (V *) malloc(n_cols * sizeof(*row));
            if ((col_no == NULL) || (row == NULL))
                ret = 2;
        }
        Vector<V> sign(slope.size());
        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (auto i: exteriorIndices) {
                for (int j = 0; j < n_cols; ++j) {
                    col_no[j] = j + 1;
                    row[j] = (Directions[i])(j);
                }
                if (!add_constraintex(lp, n_cols, row, col_no, LE, 0.)){
                    ret = 3;
                }
            }
            for (int j = 0; j < n_cols; ++j) {
                col_no[j] = j + 1;
                row[j] = sign[j];
            }
            if (!add_constraintex(lp, n_cols, row, col_no, EQ, 1.))
                ret = 3;
        }
        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            for (int j = 0; j < n_cols; ++j) {
                col_no[j] = j + 1;
                row[j] = slope[j];
            }
            if (!set_obj_fnex(lp, n_cols, row, col_no))
                ret = 4;
        }

        if (ret == 0) {
            // set the object weightVector to maximize
            set_maxim(lp);
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
        } else {
            std::cout<< "error in optimization (Ret = " << ret << ")\n";
        }

        descentDirection = VectorMap<V>(row, n_cols);
        // free allocated memory
        if (row != NULL)
            free(row);
        if (col_no != NULL)
            free(col_no);
        if (lp != NULL) {
            // clean up such that all used memory by lpsolve is freed
            delete_lp(lp);
        }

        return ret;
    }

    template class FrankWolfeOuterOptimizer<double>;
}// namespace mopmc::optimization::optimizers