//
// Created by guoxin on 3/04/24.
//

#include "ConvexQuery.h"
#include "mopmc-src/Printer.h"
#include "mopmc-src/optimizers/HalfspacesIntersection.h"
#include "mopmc-src/auxiliary/VectorConversion.h"
#include "mopmc-src/Printer.h"

namespace mopmc::queries {
    template<typename V, typename I>
    void ConvexQuery<V, I>::query() {
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<V> threshold = Eigen::Map<Vector<V>>(this->queryData.thresholds.data(), n_objs);
        Vector<V> vertex(n_objs), direction(n_objs), innerPointPrev(n_objs), outerPointPrev(n_objs);
        const V toleranceInnerOuterDiff{1.e-18}, toleranceUpdateDiff{1.e-30};
        const uint_fast64_t maxIter{200};
        V epsilonInnerOuterDiff, margin;
        this->VIhandler->initialize();

        direction.setConstant(static_cast<V>(1.0) / n_objs);// initial direction
        iter = 0;
        while (iter < maxIter) {
            std::cout << "[Main loop] Iteration: " << iter << "\n";

            /* compute a new supporting hyperplane */
            std::vector<V> direction_tmp(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction_tmp);
            std::vector<V> vertex_tmp = this->VIhandler->getResults();
            vertex = VectorMap<V>(vertex_tmp.data(), n_objs);

            Vertices.push_back(vertex);
            BoundaryPoints.push_back(vertex);
            Directions.push_back(direction);
            /* collect optimal schedulers */
            this->queryData.collectionOfSchedulers.push_back(this->VIhandler->getScheduler());

            if (Vertices.size() == 1) {
                innerPoint = vertex;
            }

            /* use the opp. direction in initial several iterations
             * as this can speed up the searching of an opt. direction */
            if (iter % 2 == 1 and iter < 10) {
                direction *= static_cast<V>(-1.0);
                ++iter;
                continue;
            }

            if (this->hasConstraint) {
                if (!mopmc::optimization::optimizers::HalfspacesIntersection<V>::findNonExteriorPoint(
                        outerPoint, BoundaryPoints, Directions)) {
                    ++iter;
                    std::cout << "[Main loop] exits as constraints are not satisfiable\n";
                    break;
                }
            } else {
                outerPoint = innerPoint;
            }

            /* This part differs from the corresponding line of the algorithm
             * in that the outer optimizer is called in every step. */
            this->outerOptimizer->minimize(outerPoint, BoundaryPoints, Directions);

            if (this->innerOptimizer->optimizeSeparationDirection(
                    direction, innerPoint, margin, Vertices, outerPoint) != EXIT_SUCCESS) {
                ++iter;
                std::cout << "[Main loop] exits as no separation hyperplane is found\n";
                break;
            }

            /* In the following termination condition, we compare the absolute difference between the inner/outer points
             * rather than the function value difference. */
            epsilonInnerOuterDiff = (innerPoint - outerPoint).template lpNorm<1>();
            // epsilonInnerOuterDiff = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (iter > 1 && epsilonInnerOuterDiff < toleranceInnerOuterDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to small inner & outer value difference (<="
                          << toleranceInnerOuterDiff << ")\n";
                break;
            }

            /* We add a new termination condition for the main loop:
             * It terminates if the improvement of inner/outer points is very small. */
            if (iter > 1 && (outerPoint - outerPointPrev).template lpNorm<1>() +
                            (innerPoint - innerPointPrev).template lpNorm<1>() < toleranceUpdateDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to small inner/outer points update (l1 norm <= " << toleranceUpdateDiff << ")\n";
                break;
            }
            outerPointPrev = outerPoint;
            innerPointPrev = innerPoint;

            /* re-normalize direction */
            direction /= direction.template lpNorm<1>();
            ++iter;
        }
        mopmc::optimization::auxiliary::VectorConversion<V>::eigenToStdVector(
                this->innerOptimizer->getVertexWeights(), this->queryData.schedulerDistribution);
        this->VIhandler->exit();
    }

    template<typename V, typename I>
    void ConvexQuery<V, I>::constraintsToHalfspaces() {
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            Vector<V> r = Vector<V>::Zero(m);
            Vector<V> w = Vector<V>::Zero(m);
            r(i) = h(i);
            w(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<V>(1.) : static_cast<V>(-1.);
            BoundaryPoints.push_back(r);
            Directions.push_back(w);
        }
    }

    template<typename V, typename I>
    bool ConvexQuery<V, I>::checkConstraintSatisfaction(const Vector<V> &point) {
        bool res = true;
        const V epsilon = 1e-8;
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            if (this->queryData.isThresholdUpperBound[i]) {
                if (point(i) > h(i) + epsilon) {
                    res = false;
                    break;
                }
            } else {
                if (point(i) < h(i) - epsilon) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    }

    template<typename V, typename I>
    void ConvexQuery<V, I>::printResult() {
        std::cout << "--Convex Query Result--"
                  << "\nwith constraint? " << std::boolalpha << hasConstraint
                  << "\nterminates after " << this->getMainLoopIterationCount() << " iteration(s)\n";
        mopmc::Printer<V>::printVector("Estimated optimal inner point", this->getInnerOptimalPoint());
        mopmc::Printer<V>::printVector("Estimated optimal outer point", this->getOuterOptimalPoint());
        std::cout << "Approximate distance (at inner point): " << this->getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << this->getOuterOptimalValue() << "\n";
        if (this->hasConstraint) {
            bool b1 = checkConstraintSatisfaction(innerPoint);
            bool b2 = checkConstraintSatisfaction(outerPoint);
            std::cout << "\nInner point satisfying constraints? " << std::boolalpha << b1
                      << "\nOuter point satisfying constraints? " << std::boolalpha << b2 << "\n";
        }
        std::cout << "----------------------------------------------\n";
    }

    template
    class ConvexQuery<double, int>;
}// namespace mopmc::queries
