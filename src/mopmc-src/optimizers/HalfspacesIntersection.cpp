//
// Created by guoxin on 30/03/24.
//

#include "HalfspacesIntersection.h"
#include "lp_lib.h"
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    int HalfspacesIntersection<V>::check(const std::vector<Vector<V>> &Vertices,
                                              const std::vector<Vector<V>> &Directions,
                                              Vector<V> &point,
                                              bool &feasible) {

        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Vertices.empty());
        n_cols = Vertices[0].size() ;// number of variables in the model
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
            for (int i = 0; i < Vertices.size(); ++i) {
                for (int j = 0; j < n_cols; ++j) {
                    col_no[j] = j + 1;
                    row[j] = Directions[i](j);
                }
                const V v = Directions[i].dot(Vertices[i]);
                if (!add_constraintex(lp, n_cols, row, col_no, LE, v))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE);// rowmode should be turned off again when done building the model
            //use a constant objective
            //col_no[0] = n_cols;
            //row[0] = 0; /* set the objective in lpsolve */
            if (!set_obj_fnex(lp, 0, row, col_no))
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
                ret = 0;
                feasible = true;
            }
            else if (ret == INFEASIBLE) {
                ret = 2;
                feasible = false;
            }
            else {
                ret = 5;
                throw std::runtime_error("Numerical Failure");
            }
        }

        if (ret == 0) {
            get_variables(lp, row);
            point = VectorMap<V>(row, n_cols);
        }
        return ret;
    }

    template class HalfspacesIntersection<double>;

}
