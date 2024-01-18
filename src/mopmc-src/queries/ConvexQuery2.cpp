//
// Created by guoxin on 16/01/24.
//

#include "ConvexQuery2.h"
#include <Eigen/Dense>
#include <iostream>

namespace mopmc::queries {

    template<typename T, typename I>
    void ConvexQuery2<T, I>::query() {
        this->VIhandler->initialize();
        const uint64_t n_objs = this->data_.objectiveCount;
        //assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        Vector<T> threshold = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), n_objs);
        std::vector<Vector<T>> Vertices, Directions;
        Vector<T> vertex(n_objs), direction(n_objs);
        Vector<T> innerPoint(n_objs), outerPoint(n_objs);
        // initial direction
        direction.setConstant(static_cast<T>(-1.0) / n_objs);
        // tolerances on exit
        const T toleranceDistanceToMinimum{1.e-6}, toleranceSmallGradient{1.e-8};
        const uint_fast64_t maxIter{200};
        T epsilonDistanceToMinimum, epsilonSmallGradient;
        uint_fast64_t iter = 0;
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
                outerPoint = vertex;
            }
            this->outerOptimizer->minimize(outerPoint, Vertices, Directions);
            epsilonDistanceToMinimum = this->fn->value(innerPoint) - this->fn->value(outerPoint);
            if (epsilonDistanceToMinimum < toleranceDistanceToMinimum) {
                std::cout << "loop exit due to small gap between inner and outer points (" << epsilonDistanceToMinimum << ")\n";
                ++iter;
                break;
            }
            ++iter;
        }
        this->VIhandler->exit();
        //printing results
        std::cout << "----------------------------------------------\n"
                  << "CUDA CONVEX QUERY terminates after " << iter << " iteration(s)\n"
                  << "Estimated nearest point to threshold : [";
        for (int i = 0; i < n_objs; ++i) {
            std::cout << innerPoint(i) << " ";
        }
        std::cout << "]\n"
                  << "Approximate distance (at inner point): " << this->fn->value(innerPoint)
                  << "\nApproximate distance (at outer point): " << this->fn->value(outerPoint)
                  << "\n----------------------------------------------\n";
    }

    template class ConvexQuery2<double, int>;
}

