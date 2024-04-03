//
// Created by guoxin on 3/04/24.
//

#include "ConvexQueryAlt.h"
#include "mopmc-src/optimizers/HalfspacesIntersectionCheck.h"

namespace mopmc::queries {
    template<typename T, typename I>
    void ConvexQueryAlt<T, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<T> threshold = Eigen::Map<Vector<T>>(this->queryData.thresholds.data(), n_objs);
        Vector<T> vertex(n_objs), direction(n_objs);
        direction.setConstant(static_cast<T>(-1.0) / n_objs);// initial direction
        const T toleranceDistanceToMinimum{1.e-6}, toleranceSmallGradient{1.e-8}, toleranceValueImpr{1.e-6}, toleranceMargin{1.e-16};
        const uint_fast64_t maxIter{130};
        T epsilonDistanceToMinimum, epsilonSmallGradient, epsilonInnerValueImpr, epsilonOuterValueImpr;
        //T innerValueCurrent, innerValueNew, outerValueCurrent, outerValueNew;
        T margin;
        bool feasible;
        constraintsToHalfspaces();
        iter = 0;
        while (iter < maxIter) {
            std::cout << "[Main loop] Iteration: " << iter << "\n";
            // compute a new supporting hyperplane
            std::vector<T> direction_tmp(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction_tmp);
            std::vector<T> vertex_tmp = this->VIhandler->getResults();
            vertex = VectorMap<T>(vertex_tmp.data(), n_objs);
            Vertices.push_back(vertex);
            Points.push_back(vertex);
            Directions.push_back(direction);
            mopmc::optimization::optimizers::HalfspacesIntersectionCheck<T>::check(Points, Directions, outerPoint, feasible);
            //std::cout << "feasible: " << feasible << ", outerPoint: " << outerPoint << "\n";
            if (!feasible) {
                ++iter;
                std::cout << "[Main loop] exits as the problem is infeasible\n";
                break;
            }
            if (Vertices.size() == 1) {
                innerPoint = vertex;
            }
            outerPoint = innerPoint;
            if (this->outerOptimizer->minimize(outerPoint, Points, Directions) != EXIT_SUCCESS) {
                break;
            };
            if (this->innerOptimizer->minimize(direction, innerPoint, margin, Vertices, outerPoint) != EXIT_SUCCESS) {
                break;
            };
            //if (margin < toleranceMargin)
            //    std::cout << "margin: " << margin <<"\n";
            epsilonDistanceToMinimum = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (epsilonDistanceToMinimum < toleranceDistanceToMinimum) {
                std::cout << "[Main loop] exit due to small gap between inner and outer points ("
                          << epsilonDistanceToMinimum << ")\n";
                ++iter;
                break;
            }
            //direction = (outerPoint - innerPoint) / (outerPoint - innerPoint).template lpNorm<1>();
            ++iter;
        }
        this->VIhandler->exit();
    }

    template<typename T, typename I>
    void ConvexQueryAlt<T, I>::constraintsToHalfspaces() {
        const uint64_t m = this->queryData.objectiveCount;
        Vector<T> h = VectorMap<T>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            Vector<T> r = Vector<T>::Zero(m);
            r(i) = h(i);
            Points.push_back(r);
            Vector<T> w = Vector<T>::Zero(m);
            w(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<T>(1.) : static_cast<T>(-1.);
            Directions.push_back(w);
        }
    }

    template<typename T, typename I>
    bool ConvexQueryAlt<T, I>::checkConstraint(const Vector<T> &point) {
        bool res = true;
        const uint64_t m = this->queryData.objectiveCount;
        Vector<T> h = VectorMap<T>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            if (this->queryData.isThresholdUpperBound[i]) {
                if (point(i) > h(i)) {
                    res = false;
                    break;
                }
            } else {
                if (point(i) < h(i)) {
                    res = false;
                    break;
                }
            }

        }
        return res;
    }

    template<typename V, typename I>
    void ConvexQueryAlt<V, I>::printResult() {
        std::cout << "----------------------------------------------\n"
                  << "CUDA CONVEX QUERY terminates after " << this->getMainLoopIterationCount() << " iteration(s)\n"
                  << "Estimated nearest point to threshold : [";
        for (int i = 0; i < this->getInnerOptimalPoint().size(); ++i) {
            std::cout << this->getInnerOptimalPoint()(i) << " ";
        }
        std::cout << "]\n"
                  << "Approximate distance (at inner point): " << this->getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << this->getOuterOptimalValue()
                  << "\nInner point satisfying constraints? " << std::boolalpha << checkConstraint(innerPoint)
                  << "\n----------------------------------------------\n";
    }

    template class ConvexQueryAlt<double, int>;
}// namespace mopmc::queries
