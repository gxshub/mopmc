//
// Created by guoxin on 30/03/24.
//

#include "ConstrainedConvexQuery.h"
#include "mopmc-src/optimizers/HalfspacesIntersectionCheck.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void ConstrainedConvexQuery<T,I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<T> threshold = Eigen::Map<Vector<T>>(this->queryData.thresholds.data(), n_objs);
        Vector<T> vertex(n_objs), direction(n_objs);
        direction.setConstant(static_cast<T>(-1.0) / n_objs);// initial direction
        const T toleranceDistanceToMinimum{1.e-6}, toleranceSmallGradient{1.e-8}, toleranceValueImpr{1.e-6};
        const uint64_t maxIter{20};
        T epsilonDistanceToMinimum, epsilonSmallGradient, epsilonInnerValueImpr, epsilonOuterValueImpr;
        T innerValueCurrent, innerValueNew, outerValueCurrent, outerValueNew;
        iter = 0;
        constraintsToHalfspaces();
        while (iter < maxIter) {
            std::cout << "[Main loop] Iteration " << iter << "\n";
            ++iter;
            // compute a supporting hyperplane
            std::vector<T> direction_tmp(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction_tmp);
            std::vector<T> vertex_tmp = this->VIhandler->getResults();
            vertex = VectorMap<T>(vertex_tmp.data(), n_objs);
            Vertices.push_back(vertex);
            Points.push_back(vertex);
            Directions.push_back(direction);

            mopmc::optimization::optimizers::HalfspacesIntersectionCheck<T>::check(Points, Directions, outerPoint, feasible);
            //std::cout << "feasible: " << feasible << ", outerPoint: " << outerPoint << "\n";
            if (!feasible)
                break;
            this->outerOptimizer->minimize(outerPoint, Points, Directions);
            this->innerOptimizer->minimize(innerPoint, Vertices, outerPoint);
            epsilonDistanceToMinimum = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (checkConstraint(innerPoint) && epsilonDistanceToMinimum < toleranceDistanceToMinimum) {
                std::cout << "[Main loop] exit due to small gap between inner and outer points ("
                          << epsilonDistanceToMinimum << ")\n";
                break;
            }
            innerValueNew = this->fn->value(innerPoint);
            outerValueNew = this->fn->value(outerPoint);
            std::cout << "[Main loop] innerValueNew: " << innerValueNew <<", outerValueNew: " << outerValueNew <<"\n";
            epsilonInnerValueImpr = (innerValueCurrent - innerValueNew) / std::max(std::abs(innerValueCurrent), 1.);
            epsilonOuterValueImpr = (outerValueNew - outerValueCurrent) / std::max(std::abs(innerValueCurrent), 1.);
            if (iter >= 2 && std::max(epsilonInnerValueImpr, epsilonOuterValueImpr) < toleranceValueImpr) {
                std::cout << "[Main loop] exit due to small relative improvement on (estimated) nearest points ("
                          << std::max(epsilonInnerValueImpr, epsilonOuterValueImpr) << ")\n";
                break;
            }

            innerValueCurrent = innerValueNew;
            outerValueCurrent = outerValueNew;

            Vector<T> grad = this->fn->subgradient(innerPoint);
            epsilonSmallGradient = grad.template lpNorm<1>();
            if (epsilonSmallGradient < toleranceSmallGradient) {
                std::cout << "[Main loop] exit due to small gradient (" << epsilonSmallGradient << ")\n";
                break;
            }
            direction = (innerPoint - outerPoint)/ (innerPoint - outerPoint).template lpNorm<1>();
            std::cout << "[Main loop] direction: " << direction <<"\n";
        }
        this->VIhandler->exit();
    }

    template<typename T, typename I>
    void ConstrainedConvexQuery<T,I>::constraintsToHalfspaces() {
        const uint64_t m = this->queryData.objectiveCount;
        Vector<T> h = VectorMap<T>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            Vector<T> r = Vector<T>::Zero(m);
            r(i) = h(i);
            Points.push_back(r);
            Vector<T> w = Vector<T>::Zero(m);
            w(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<T>(1.) : static_cast<T>(0.);
            Directions.push_back(w);
        }

    }

    template<typename T, typename I>
    bool ConstrainedConvexQuery<T,I>::checkConstraint(const Vector<T> &point) {
        bool res = true;
        const uint64_t m = this->queryData.objectiveCount;
        Vector<T> h = VectorMap<T>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            if (this->queryData.isThresholdUpperBound[i]) {
                if (point(i) > h(i)) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    }

    template<typename V, typename I>
    void ConstrainedConvexQuery<V, I>::printResult() {
        std::cout << "----------------------------------------------\n"
                  << "CUDA CONVEX QUERY terminates after " << this->getMainLoopIterationCount() << " iteration(s)\n"
                  << "Estimated nearest point to threshold : [";
        for (int i = 0; i < this->getInnerOptimalPoint().size(); ++i) {
            std::cout << this->getInnerOptimalPoint()(i) << " ";
        }
        bool b = checkConstraint(innerPoint);
        std::cout << "]\n"
                  << "Approximate distance at inner point: " << this->getInnerOptimalValue()
                  << "\nApproximate distance at outer point: " << this->getOuterOptimalValue()
                  <<"\nInner point satisfying constraints? " << std::boolalpha  << b
                  << "\n----------------------------------------------\n";
    }

    template class ConstrainedConvexQuery<double, int>;
}// namespace mompc::queries
