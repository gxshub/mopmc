//
// Created by guoxin on 3/04/24.
//

#include "ConvexQuery.h"
#include "mopmc-src/Printer.h"
#include "mopmc-src/optimizers/HalfspacesIntersection.h"
#include "mopmc-src/auxiliary/VectorConversion.h"

namespace mopmc::queries {
    template<typename V, typename I>
    void ConvexQuery<V, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->queryData.objectiveCount;
        Vector<V> threshold = Eigen::Map<Vector<V>>(this->queryData.thresholds.data(), n_objs);
        Vector<V> vertex(n_objs), direction(n_objs), innerPointPrev(n_objs), outerPointPrev(n_objs);
        direction.setConstant(static_cast<V>(-1.0) / n_objs);// initial direction
        const V toleranceInnerOuterDiff{1.e-18}, toleranceUpdateDiff{1.e-16};
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
            this->queryData.collectionOfSchedulers.push_back(this->VIhandler->getScheduler());
            if (Vertices.size() == 1)
                innerPoint = vertex;
            if (iter % 2 == 0) {
                //use the opp direction in each other iteration
                const V c = direction.template lpNorm<1>();
                direction /= -c;
                ++iter;
                continue;
            }
            if (this->hasConstraint) {
                if (!mopmc::optimization::optimizers::HalfspacesIntersection<V>::findNonExteriorPoint(
                        outerPoint, BoundaryPoints, Directions)) {
                    ++iter;
                    std::cout << "[Main loop] exits as the constraint is not satisfiable\n";
                    break;
                }
            } else {
                outerPoint = innerPoint;
            }
            this->outerOptimizer->minimize(outerPoint, BoundaryPoints, Directions);
            if (iter > 1 && (outerPoint - outerPointPrev).template lpNorm<1>() +
                            (innerPoint - innerPointPrev).template lpNorm<1>() <
                            toleranceUpdateDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to inner/outer points update (l1 norm <= " << toleranceUpdateDiff << ")\n";
                break;
            }
            outerPointPrev = outerPoint;
            innerPointPrev = innerPoint;
            if (this->innerOptimizer->optimizeSeparationDirection(
                    direction, innerPoint, margin, Vertices, outerPoint) != EXIT_SUCCESS) {
                ++iter;
                std::cout << "[Main loop] exits as no separation hyperplane is found\n";
                //break;
            }
            // re-normalize direction if necessary
            direction /= direction.template lpNorm<1>();
            epsilonInnerOuterDiff = (innerPoint - outerPoint).template lpNorm<1>();
            // epsilonInnerOuterDiff = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (iter > 1 && epsilonInnerOuterDiff < toleranceInnerOuterDiff) {
                ++iter;
                std::cout << "[Main loop] exits due to small inner & outer value difference (<="
                          << toleranceInnerOuterDiff << ")\n";
                break;
            }
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
        const V errorTorPc = 0.001;
        const uint64_t m = this->queryData.objectiveCount;
        Vector<V> h = VectorMap<V>(this->queryData.thresholds.data(), m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            if (this->queryData.isThresholdUpperBound[i]) {
                if (point(i) > h(i) * (1 + errorTorPc)) {
                    res = false;
                    break;
                }
            } else {
                if (point(i) < h(i) * (1 - errorTorPc)) {
                    res = false;
                    break;
                }
            }
        }
        mopmc::Printer<V>::printVector("check constraint satisfaction: ", point - h);
        return res;
    }

    template<typename V, typename I>
    void ConvexQuery<V, I>::printResult() {
        std::cout << "----------------------------------------------\n"
                  << "--Convex Query Result--"
                  << "\nwith constraint? " << std::boolalpha << hasConstraint
                  << "\nterminates after " << this->getMainLoopIterationCount() << " iteration(s)\n";
        mopmc::Printer<V>::printVector("Estimated optimal outer point", this->getOuterOptimalPoint());
        std::cout << "Approximate distance (at inner point): " << this->getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << this->getOuterOptimalValue() << "\n";
        if (this->hasConstraint) {
            bool b1 = checkConstraintSatisfaction(innerPoint);
            bool b2 = checkConstraintSatisfaction(outerPoint);
            std::cout << "\nInner point satisfying constraints? " << std::boolalpha << b1
                      << "\nOuter point satisfying constraints? " << std::boolalpha << b2 << "\n";
        }
        //mopmc::Printer<V>::printVector("\n[debug] scheduler distribution", this->queryData.schedulerDistribution);
        //std::cout << "\n[debug] sum of inner point: " << this->getInnerOptimalPoint().sum();
        //std::cout << "\n[debug] sum of outer point: " << this->getOuterOptimalPoint().sum();
        std::cout << "----------------------------------------------\n";
    }

    template
    class ConvexQuery<double, int>;
}// namespace mopmc::queries
