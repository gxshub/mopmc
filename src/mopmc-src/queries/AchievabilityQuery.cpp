//
// Created by guoxin on 1/12/23.
//


#include "AchievabilityQuery.h"
#include "mopmc-src/optimizers/MaximumMarginSeparationHyperplane.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void AchievabilityQuery<T, I>::query() {
        assert(this->queryData.rowGroupIndices.size() == this->queryData.colCount + 1);
        mopmc::optimization::optimizers::MaximumMarginSeparationHyperplane<T> separationHyperplaneOptimizer;
        this->VIhandler->initialize();
        const uint64_t nObjs = this->queryData.objectiveCount;
        Vector<T> threshold = Eigen::Map<Vector<T>>(this->queryData.thresholds.data(), this->queryData.thresholds.size());
        Vector<T> sign(nObjs);
        for (uint_fast64_t i=0; i< sign.size(); ++i) {
            sign(i) = this->queryData.isThresholdUpperBound[i] ? static_cast<T>(-1) : static_cast<T>(1);
        }
        Vector<T> vertex(nObjs), weightVector(nObjs);
        std::vector<double> vertex_(nObjs + 1), weightVector_(nObjs);
        const uint64_t maxIter{20};
        iter = 0;
        weightVector.setConstant(static_cast<T>(1.0) / nObjs); //initial weightVector
        achievable = true;
        T delta;

        while (iter < maxIter) {
            if (!VertexVectors.empty()) {
                separationHyperplaneOptimizer.findMaximumSeparatingDirection(VertexVectors,
                                                                             threshold,
                                                                             sign,
                                                                             weightVector,
                                                                             delta);
                if (delta <= 0)
                    break;
            }
            for (uint_fast64_t i = 0; i < weightVector.size(); ++i) {
                weightVector_[i] = (double) (sign(i) * weightVector(i));
            }
            this->VIhandler->valueIteration(weightVector_);
            vertex_ = this->VIhandler->getResults();
            for (uint_fast64_t i = 0; i < nObjs; ++i) {
                vertex(i) = (T) vertex_[i];
            }
            VertexVectors.push_back(vertex);
            WeightVectors.push_back(weightVector);

            Vector<T> wTemp = (sign.array() * weightVector.array()).matrix();
            if (wTemp.dot(threshold - vertex) > 0) {
                achievable = false;
                ++iter;
                break;
            }
            ++iter;
        }
        this->VIhandler->exit();
    }


    template<typename T, typename I>
    void AchievabilityQuery<T, I>::printResult() {
        std::cout << "----------------------------------------------\n";
        std::cout << "Achievability Query terminates after " << this->getMainLoopIterationCount() << " iteration(s) \n";
        std::cout << "OUTPUT: " << std::boolalpha << this->getResult() << "\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class AchievabilityQuery<double, int>;
}