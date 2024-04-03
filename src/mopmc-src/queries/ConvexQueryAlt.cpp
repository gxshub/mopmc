//
// Created by guoxin on 3/04/24.
//

#include "ConvexQueryAlt.h"
#include "mopmc-src/optimizers/HalfspacesIntersectionCheck.h"

namespace mopmc::queries {
    template<typename V, typename I>
    void ConvexQueryAlt<V, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<V> threshold = Eigen::Map<Vector<V>>(this->queryData.thresholds.data(), n_objs);
        Vector<V> vertex(n_objs), direction(n_objs);
        direction.setConstant(static_cast<V>(-1.0) / n_objs);// initial direction
        const V toleranceDistanceToMinimum{1.e-6};
        const uint_fast64_t maxIter{130};
        V epsilonInnerOuterDiff;
        V margin;
        bool feasible;
        constraintsToHalfspaces();
        iter = 0;
        while (iter < maxIter) {
            std::cout << "[Main loop] Iteration: " << iter << "\n";
            // compute a new supporting hyperplane
            std::vector<V> direction_tmp(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction_tmp);
            std::vector<V> vertex_tmp = this->VIhandler->getResults();
            vertex = VectorMap<V>(vertex_tmp.data(), n_objs);
            Vertices.push_back(vertex);
            Points.push_back(vertex);
            Directions.push_back(direction);
            mopmc::optimization::optimizers::HalfspacesIntersectionCheck<V>::check(Points, Directions, outerPoint, feasible);
            if (!feasible) {
                ++iter;
                std::cout << "[Main loop] exits as the problem is infeasible\n";
                break;
            }
            this->outerOptimizer->minimize(outerPoint, Points, Directions);
            if (Vertices.size() == 1)
                innerPoint = vertex;
            if (this->innerOptimizer->minimize(direction, innerPoint, margin, Vertices, outerPoint) != EXIT_SUCCESS)
                break;
            epsilonInnerOuterDiff = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (iter > 1 && epsilonInnerOuterDiff < toleranceDistanceToMinimum) {
                std::cout << "[Main loop] exit due to small gap between inner and outer points ("
                          << epsilonInnerOuterDiff << ")\n";
                ++iter;
                break;
            }
            ++iter;
        }
        this->VIhandler->exit();
    }

    template<typename V, typename I>
    void ConvexQueryAlt<V, I>::constraintsToHalfspaces() {
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            Vector<V> r = Vector<V>::Zero(m);
            r(i) = h(i);
            Points.push_back(r);
            Vector<V> w = Vector<V>::Zero(m);
            w(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<V>(1.) : static_cast<V>(-1.);
            Directions.push_back(w);
        }
    }

    template<typename V, typename I>
    bool ConvexQueryAlt<V, I>::checkConstraint(const Vector<V> &point) {
        bool res = true;
        const V roundingErr = 1e-18;
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            if (this->queryData.isThresholdUpperBound[i]) {
                if (point(i) > h(i) + roundingErr) {
                    res = false;
                    break;
                }
            } else {
                if (point(i) < h(i) - roundingErr) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    }

    template<typename V, typename I>
    void ConvexQueryAlt<V, I>::printResult() {
        bool b = checkConstraint(outerPoint);
        std::cout << "----------------------------------------------\n"
                  << "CUDA CONVEX QUERY terminates after " << this->getMainLoopIterationCount() << " iteration(s)\n"
                  << "Estimated nearest point to threshold : [";
        for (int i = 0; i < this->queryData.objectiveCount; ++i) {
            std::cout << this->getOuterOptimalPoint()(i) << " ";
        }
        std::cout << "]\n"
                  << "Approximate distance (at inner point): " << this->getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << this->getOuterOptimalValue()
                  << "\nOptimal point satisfying constraints? " << std::boolalpha << b
                  << "\n----------------------------------------------\n";
    }

    template class ConvexQueryAlt<double, int>;
}// namespace mopmc::queries
