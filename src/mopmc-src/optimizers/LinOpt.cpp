//
// Created by guoxin on 24/11/23.
//

#include <iostream>
#include "LinOpt.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    int LinOpt<V>::findMaximumFeasibleStep(const std::vector<Vector<V>> &Phi,
                                           const Vector<V> &d,
                                           Vector<V> point,
                                           V step) {

        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Phi.empty());
        n_cols = Phi.size() + 1; // number of variables in the model
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

        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[j] = j + 1;
                row[j] = static_cast<V>(1.);
            }
            if (!add_constraintex(lp, n_cols - 1, row, col_no, EQ, static_cast<V>(1.)))
                ret = 3;
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[0] = j + 1;
                row[0] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
            for (int i = 0; i < Phi[0].size(); ++ i) {
                for (int j = 0; j < n_cols - 1; ++j) {
                    col_no[j] = j + 1;
                    row[j] = Phi[j][i];
                }
                col_no[n_cols-1] = n_cols;
                row[n_cols-1] = d(i);
                if (!add_constraintex(lp, n_cols, row, col_no, EQ, point(i)))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            col_no[0] =  n_cols;
            row[0] = 1;
            if (!set_obj_fnex(lp, 1, row, col_no))
                ret = 4;
        }

        if (ret == 0) {
            set_minim(lp);
            //write_LP(lp, stdout);
            // write_lp(lp, "model.lp");
            set_verbose(lp, IMPORTANT);
            ret = solve(lp);
            //printf("ret = %i\n", ret);
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
            throw std::runtime_error("runtime error in linopt");
        }
        //result
        step = row[n_cols-1];
        //std::cout << "step: " << step <<"\n";

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
    };

    template<typename V>
    int LinOpt<V>::checkPointInConvexHull(const std::vector<Vector<V>> &Vertices,
                                          const Vector<V> &point,
                                          int &feasible){
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Vertices.empty());
        n_cols = Vertices.size(); // number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1; // couldn't construct a new model

        set_break_numeric_accuracy(lp, 1e-6);
        set_scaling(lp, SCALE_CURTISREID);
        if (ret == 0) {
            // create space large enough for one row
            col_no = (int *) malloc(n_cols * sizeof(*col_no));
            row = (V *) malloc(n_cols * sizeof(*row));
            if ((col_no == NULL) || (row == NULL))
                ret = 2;
        }

        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (int j = 0; j < n_cols; ++j) {
                col_no[j] = j + 1;
                row[j] = static_cast<V>(1.);
            }
            if (!add_constraintex(lp, n_cols, row, col_no, EQ, static_cast<V>(1.)))
                ret = 3;
            /*
             * All variables in lp solve are >=0 by default.
            for (int j = 0; j < n_cols; ++j) {
                col_no[0] = j + 1;
                row[0] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
             */
            for (int i = 0; i < Vertices[0].size(); ++ i) {
                for (int j = 0; j < n_cols; ++j) {
                    col_no[j] = j + 1;
                    row[j] = Vertices[j][i];
                }
                if (!add_constraintex(lp, n_cols, row, col_no, EQ, point[i]))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            col_no[0] = 0;
            row[0] = 0;
            if (!set_obj_fnex(lp, 1, row, col_no))
                ret = 4;
        }

        if (ret == 0) {
            set_minim(lp);
            //write_LP(lp, stdout);
            // write_lp(lp, "model.lp");
            set_verbose(lp, IMPORTANT);
            ret = solve(lp);
            //printf("ret = %i\n", ret);
            // ----copy result here----
            feasible = ret;
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
            //std::cout << "get optimal solution~\n";
            // we are done now
        } else {
            //std::cout<< "error in optimization (Ret = " << ret << ")\n";
        }


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

    template<typename V>
    int LinOpt<V>::findOptimalSeparatingDirection(const std::vector<Vector<V>> &Phi,
                                                  const Vector<V> &d,
                                                  const Vector<V> &sgn,
                                                  Vector<V> &weightVector,
                                                  V &gap) {

        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Phi.empty());
        n_cols = Phi[0].size() + 1; // number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1; // couldn't construct a new model
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
            for (int i = 0; i < Phi.size(); ++i) {
                for (int j = 0; j < n_cols - 1; ++j) {
                    col_no[j] = j + 1;
                    row[j] = sgn(j) * (d(j) - Phi[i](j));
                    //std::cout << "row[j]: " << row[j] <<"\n";
                }
                col_no[n_cols - 1] = n_cols;
                row[n_cols - 1] = static_cast<V>(-1.);
                if (!add_constraintex(lp, n_cols, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            col_no[0] = n_cols;
            row[0] = static_cast<V>(1.);/* set the objective in lpsolve */
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
            std::cout<< "error in optimization (Ret = " << ret << ")\n";
        }

        weightVector = VectorMap<V>(row, n_cols - 1);
        gap = row[n_cols - 1];
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

    template<typename V>
    int LinOpt<V>::findOptimalProjectedDescentDirection(const std::vector<Vector<V>> &Directions,
                                                        const std::vector<uint64_t> &exteriorIndices,
                                                        const Vector<V> &gradient,
                                                        Vector<V> &descentDirection){
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!exteriorIndices.empty());
        n_cols = (int) gradient.size(); // number of variables in the model
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
        Vector<V> sign(gradient.size());
        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (auto i: exteriorIndices) {
                for (int j = 0; j < n_cols; ++j) {
                    col_no[j] = j + 1;
                    row[j] = (Directions[i])(j);
                    if (!add_constraintex(lp, n_cols, row, col_no, LE, 0.)){
                        ret = 3;
                    }
                }
            }
            for (int j = 0; j < n_cols; ++j) {
                col_no[j] = j + 1;
                row[j] = sign[j];
            }
            if (!add_constraintex(lp, n_cols - 1, row, col_no, EQ, 1.))
                ret = 3;
        }
        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            for (int j = 0; j < n_cols; ++j) {
                col_no[j] = j + 1;
                row[j] = gradient[j];
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

    template
    class LinOpt<double>;
}