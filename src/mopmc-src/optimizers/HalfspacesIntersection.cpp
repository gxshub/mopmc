//
// Created by guoxin on 30/03/24.
//

#include "HalfspacesIntersection.h"
#include "mopmc-src/Printer.h"
#include "lp_lib.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace mopmc::optimization::optimizers {

    template<typename V>
    bool HalfspacesIntersection<V>::findNonExteriorPoint(Vector<V> &point,
                                                         const std::vector<Vector<V>> &BoundaryPoints,
                                                         const std::vector<Vector<V>> &Directions) {

        bool feasible(false);

        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!BoundaryPoints.empty());
        n_cols = BoundaryPoints[0].size() ;// number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1;// couldn't construct a new model
        if (ret == 0) {
            //[important!] set the unbounded variables.
            // The default bounds are >=0 in lp solve.
            // set_unbounded(lp, n_cols);
            for (int i = 1; i <= n_cols; ++i) {
                set_unbounded(lp, i);
            }
            // create space large enough for one row
            col_no = (int *) malloc(n_cols * sizeof(*col_no));
            row = (V *) malloc(n_cols * sizeof(*row));
            if ((col_no == NULL) || (row == NULL))
                ret = 2;
        }

        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (int i = 0; i < BoundaryPoints.size(); ++i) {
                for (int j = 0; j < n_cols; ++j) {
                    col_no[j] = j + 1;
                    row[j] = Directions[i](j);
                }
                const V v = Directions[i].dot(BoundaryPoints[i]);
                if (!add_constraintex(lp, n_cols, row, col_no, LE, v))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE);// rowmode should be turned off again when done building the model
            //use a constant objective
            /* set the objective in lpsolve */
            std::vector<REAL> zero_obj_coes(n_cols);
            std::fill(zero_obj_coes.begin(), zero_obj_coes.end(), static_cast<REAL>(0.0)); // All zeros
            if (!set_obj_fnex(lp, n_cols, zero_obj_coes.data(), col_no))
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
            if (ret == OPTIMAL) {
                feasible = true;
            }
            else if (ret == INFEASIBLE) {
                feasible = false;
            }
            else {
                throw std::runtime_error("Numerical Failure");
            }
        }

        if (ret == 0) {
            std::vector<V> solution_vars(n_cols);
            get_variables(lp, solution_vars.data());
            point = VectorMap<V>(solution_vars.data(), n_cols);
        }
        return feasible;
    }

    template<typename V>
    bool HalfspacesIntersection<V>::verifyPointInHalfspaces(
            const Vector<V>& point,
            const std::vector<Vector<V>>& BoundaryPoints,
            const std::vector<Vector<V>>& Directions,
            V epsilon) { // Use a small epsilon for floating-point comparisons

        // Basic sanity check: ensure sizes match
        if (BoundaryPoints.size() != Directions.size()) {
            std::cerr << "Error: BoundaryPoints and Directions vectors must have the same size." << std::endl;
            return false;
        }
        if (point.size() == 0 || BoundaryPoints.empty() || BoundaryPoints[0].size() == 0) {
            // If there are no halfspaces or the point/vectors are empty, it's trivially true or an error state.
            // Adapt this based on your exact definition of "empty intersection" or "valid input".
            return true; // No constraints means it's feasible within the whole space.
        }
        for (size_t i = 0; i < BoundaryPoints.size(); ++i) {
            // Calculate w_i . x
            V dot_wx = Directions[i].dot(point);
            // Calculate w_i . r_i
            V dot_wr = Directions[i].dot(BoundaryPoints[i]);
            if (dot_wx > dot_wr + epsilon) {
                std::cerr << "Verification failed for half-space " << i << ":" << std::endl;
                mopmc::Printer<V>::printVector("  Direction ",  Directions[i]);
                mopmc::Printer<V>::printVector("  Boundary Point ", BoundaryPoints[i]);
                mopmc::Printer<V>::printVector("  Test Point  ", point);
                std::cout << "  w.x = " << dot_wx << std::endl;
                std::cout << "  w.r = " << dot_wr << std::endl;
                std::cout << "  Violation: w.x (" << dot_wx << ") > w.r (" << dot_wr << ") + epsilon (" << epsilon << ")" << std::endl;
                return false; // Point is *not* in this half-space, so it's not in the intersection.
            }
        }

        return true; // All half-spaces satisfied
    }

    template<typename V>
    bool HalfspacesIntersection<V>::checkNonExteriorPoint(Vector<V> &point,
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
            std::cout << "[Check Non Exterior Point] Directions.size(): " << Directions.size()<<"\n";
            std::cout << "[Check Non Exterior Point] exterior point - Directions[i].dot(point) - Directions[i].dot(BoundaryPoints[i]): "
                      << Directions[i].dot(point) - Directions[i].dot(BoundaryPoints[i]) <<"\n";
        }
        return nonExterior;
    }

    template class HalfspacesIntersection<double>;

}
