//
// Created by guoxin on 2/02/24.
//

#include "SeparationHyperplaneOptimizer.h"
#include "lp_lib.h"
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int SeparationHyperplaneOptimizer<V>::findMaximumSeparatingDirection(const std::vector<Vector<V>> &Vertices,
                                                                         const Vector<V> &gradient,
                                                                         const Vector<V> &sign,
                                                                         Vector<V> &direction,
                                                                         V &distance) {
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Vertices.empty());
        n_cols = Vertices[0].size() + 1;// number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1;// couldn't construct a new model
        if (ret == 0) {
            //[important!] set the unbounded variables.
            // The default bounds are >=0 in lp solve.
            set_unbounded(lp, n_cols);
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
            if (!add_constraintex(lp, n_cols - 1, row, col_no, GE, static_cast<V>(1.)))
                ret = 3;
            if (!add_constraintex(lp, n_cols - 1, row, col_no, LE, static_cast<V>(1.)))
                ret = 3;
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[0] = j + 1;
                row[0] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
            for (int i = 0; i < Vertices.size(); ++i) {
                for (int j = 0; j < n_cols - 1; ++j) {
                    col_no[j] = j + 1;
                    row[j] = sign(j) * (gradient(j) - Vertices[i](j));
                    //std::cout << "row[j]: " << row[j] <<"\n";
                }
                col_no[n_cols - 1] = n_cols;
                row[n_cols - 1] = static_cast<V>(-1.);
                if (!add_constraintex(lp, n_cols, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE);// rowmode should be turned off again when done building the model
            col_no[0] = n_cols;
            row[0] = static_cast<V>(1.); /* set the objective in lpsolve */
            if (!set_obj_fnex(lp, 1, row, col_no))
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
            /*
            std::cout << "Optimal solutions: ";
            for (int j = 0; j < n_cols; j++)
                std::cout << get_col_name(lp, j + 1) << ": " << row[j] << ", ";
            std::cout << "\n";
             */
            // we are done now
        } else {
            std::cout << "error in optimization (Ret = " << ret << ")\n";
        }

        direction = VectorMap<V>(row, n_cols - 1);
        distance = row[n_cols - 1];
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

    template class SeparationHyperplaneOptimizer<double>;

}// namespace mopmc::optimization::optimizers