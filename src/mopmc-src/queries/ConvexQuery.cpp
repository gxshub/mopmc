//
// Created by guoxin on 3/04/24.
//

#include "ConvexQuery.h"
#include "mopmc-src/optimizers/HalfspacesIntersection.h"
#include "mopmc-src/Printer.h"

namespace mopmc::queries {
    template<typename V, typename I>
    void ConvexQuery<V, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<V> threshold = Eigen::Map<Vector<V>>(this->queryData.thresholds.data(), n_objs);
        Vector<V> vertex(n_objs), direction(n_objs), directionOps, innerPoint1(n_objs), outerPoint1(n_objs);
        direction.setZero();
        direction(0) = static_cast<V>(1.0);
        //direction.setConstant(static_cast<V>(-1.0) / n_objs);// initial direction
        const V toleranceInnerOuterDiff{1.e-18}, toleranceUpdateDiff{1.e-18};
        const uint_fast64_t maxIter{200};
        V epsilonInnerOuterDiff;
        V margin;
        iter = 0;
        while (iter < maxIter) {
            std::cout << "[Main loop] Iteration: " << iter << "\n";
            // compute a new supporting hyperplane
            std::vector<V> direction_tmp(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction_tmp);
            std::vector<V> vertex_tmp = this->VIhandler->getResults();
            vertex = VectorMap<V>(vertex_tmp.data(), n_objs);
            Vertices.push_back(vertex);
            BoundaryPoints.push_back(vertex);
            Directions.push_back(direction);
            if (Vertices.size() == 1)
                innerPoint = vertex;
            ++iter;
            directionOps = static_cast<V>(-1.) * direction;
            std::vector<V> direction_tmp_ops(directionOps.data(), directionOps.data() + directionOps.size());
            this->VIhandler->valueIteration(direction_tmp_ops);
            std::vector<V> vertex_tmp_ops = this->VIhandler->getResults();
            vertex = VectorMap<V>(vertex_tmp_ops.data(), n_objs);
            Vertices.push_back(vertex);
            BoundaryPoints.push_back(vertex);
            Directions.push_back(directionOps);
            if (this->hasConstraint) {
                if (!mopmc::optimization::optimizers::HalfspacesIntersection<V>::findNonExteriorPoint(outerPoint, BoundaryPoints, Directions)) {
                    ++iter;
                    std::cout << "[Main loop] exits as the constraint is not satisfiable\n";
                    break;
                }
            } else {
                outerPoint = innerPoint;
            }
            this->outerOptimizer->minimize(outerPoint, BoundaryPoints, Directions);
            //mopmc::Printer<V>::printVector("outerPoint (after gradient decent)", outerPoint);
            if (iter > 1 && (this->getOuterOptimalPoint() - outerPoint1).template lpNorm<1>()
                    + (this->getInnerOptimalPoint() - innerPoint1).template lpNorm<1>() < toleranceUpdateDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to outer point update (l1 norm <= " << toleranceUpdateDiff << ")\n";
                break;
            }
            outerPoint1 = outerPoint;
            innerPoint1 = innerPoint;
            if (this->innerOptimizer->optimizeSeparationDirection(direction, innerPoint, margin, Vertices, outerPoint) != EXIT_SUCCESS) {
                ++iter;
                std::cout << "[Main loop] exits as no separation hyperplane is found\n";
                break;
            }
            // re-normalize direction if necessary
            direction /= direction.template lpNorm<1>();
            epsilonInnerOuterDiff = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (iter > 1 && epsilonInnerOuterDiff < toleranceInnerOuterDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to small inner & outer value difference (<=" << toleranceInnerOuterDiff << ")\n";
                break;
            }
            ++iter;
        }
        this->VIhandler->exit();
    }

    template<typename V, typename I>
    void ConvexQuery<V, I>::constraintsToHalfspaces() {
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            Vector<V> r = Vector<V>::Zero(m);
            r(i) = h(i);
            BoundaryPoints.push_back(r);
            Vector<V> w = Vector<V>::Zero(m);
            w(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<V>(1.) : static_cast<V>(-1.);
            Directions.push_back(w);
        }
    }

    template<typename V, typename I>
    bool ConvexQuery<V, I>::checkConstraintSatisfaction(const Vector<V> &point) {
        bool res = true;
        const V roundingErr = 1e-15;
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
    void ConvexQuery<V, I>::printResult() {
        std::cout << "----------------------------------------------\n"
                  << "CONVEX QUERY"
                  << "\nwith constraint? " << std::boolalpha << hasConstraint
                  << "\nterminates after " << this->getMainLoopIterationCount() << " iteration(s)\n";
        mopmc::Printer<V>::printVector("Estimated optimal outer point", this->getOuterOptimalPoint());
        std::cout << "Approximate distance (at inner point): " << this->getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << this->getOuterOptimalValue();
        if (this->hasConstraint) {
            bool b1 = checkConstraintSatisfaction(innerPoint);
            bool b2 = checkConstraintSatisfaction(outerPoint);
            std::cout << "\nInner point satisfying constraints? " << std::boolalpha << b1
                      << "\nOuter point satisfying constraints? " << std::boolalpha << b2;
        };
        std::cout << "\n----------------------------------------------\n";
    }

    template class ConvexQuery<double, int>;
}// namespace mopmc::queries
