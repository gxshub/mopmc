//
// Created by guoxin on 16/01/24.
//

#include "ConvexQuery.h"
#include <Eigen/Dense>
#include <iostream>

namespace mopmc::queries {

    template<typename T, typename I>
    void ConvexQuery<T, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->data_.objectiveCount;
        Vector<T> threshold = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), n_objs);
        Vector<T> vertex(n_objs), direction(n_objs);
        direction.setConstant(static_cast<T>(-1.0) / n_objs);// initial direction
        const T toleranceDistanceToMinimum{1.e-6}, toleranceSmallGradient{1.e-8}, toleranceValueImpr{1.e-6};
        const uint_fast64_t maxIter{100};
        T epsilonDistanceToMinimum, epsilonSmallGradient, epsilonInnerValueImpr, epsilonOuterValueImpr;
        T innerValueCurrent, innerValueNew, outerValueCurrent, outerValueNew;
        iter = 0;
        while (iter < maxIter) {
            std::cout << "Main loop: Iteration " << iter << "\n";
            if (!Vertices.empty()) {
                this->innerOptimizer->minimize(innerPoint, Vertices);
                Vector<T> grad = this->fn->subgradient(innerPoint);
                epsilonSmallGradient = grad.template lpNorm<1>();
                if (epsilonSmallGradient < toleranceSmallGradient) {
                    std::cout << "loop exit due to small gradient (" << epsilonSmallGradient << ")\n";
                    ++iter;
                    break;
                }
                direction = static_cast<T>(-1.) * grad / grad.template lpNorm<1>();
            }
            // compute a new supporting hyperplane
            std::vector<T> direction1(direction.data(), direction.data() + direction.size());
            this->VIhandler->valueIteration(direction1);
            std::vector<T> vertex1 = this->VIhandler->getResults();
            vertex = VectorMap<T>(vertex1.data(), n_objs);
            Vertices.push_back(vertex);
            Directions.push_back(direction);
            if (Vertices.size() == 1) {
                innerPoint = vertex;
            }
            outerPoint = innerPoint;
            this->outerOptimizer->minimize(outerPoint, Vertices, Directions);
            epsilonDistanceToMinimum = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (epsilonDistanceToMinimum < toleranceDistanceToMinimum) {
                std::cout << "loop exit due to small gap between inner and outer points ("
                          << epsilonDistanceToMinimum << ")\n";
                ++iter;
                break;
            }
            innerValueNew = this->fn->value(innerPoint);
            outerValueNew = this->fn->value(outerPoint);
            epsilonInnerValueImpr = (innerValueCurrent - innerValueNew) / std::max(std::abs(innerValueCurrent), 1.);
            epsilonOuterValueImpr = (outerValueNew - outerValueCurrent) / std::max(std::abs(innerValueCurrent), 1.);
            if (iter >= 10 && std::max(epsilonInnerValueImpr, epsilonOuterValueImpr) < toleranceValueImpr) {
                std::cout << "loop exit due to small relative improvement on (estimated) nearest points ("
                          << std::max(epsilonInnerValueImpr, epsilonOuterValueImpr) << ")\n";
                ++iter;
                break;
            }
            innerValueCurrent = innerValueNew;
            outerValueCurrent = outerValueNew;
            ++iter;
        }
        this->VIhandler->exit();
    }

    template class ConvexQuery<double, int>;
}// namespace mopmc::queries
