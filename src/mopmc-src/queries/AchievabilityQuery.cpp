//
// Created by guoxin on 1/12/23.
//


#include "AchievabilityQuery.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void AchievabilityQuery<T, I>::query() {
        assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        mopmc::optimization::optimizers::LinOpt<T> linOpt;
        PolytopeType rep = Closure;
        this->VIhandler->initialize();
        const uint64_t nObjs = this->data_.objectiveCount;
        Vector<T> thresholds = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());
        std::vector<Vector<T>> Vertices, Directions;

        Vector<T> sgn(nObjs); // optimisation direction
        for (uint_fast64_t i=0; i<sgn.size(); ++i) {
            sgn(i) = this->data_.isThresholdUpperBound[i] ? static_cast<T>(-1) : static_cast<T>(1);
        }

        Vector<T> vertex(nObjs), direction(nObjs), w1(nObjs + 1);
        std::vector<double> r_(nObjs + 1), w_(nObjs);

        const uint64_t maxIter{20};
        uint_fast64_t iter = 0;
        direction.setConstant(static_cast<T>(1.0) / nObjs); //initial direction
        bool achievable = true;
        T delta;

        while (iter < maxIter) {
            if (!Vertices.empty()) {
                linOpt.findOptimalSeparatingDirection(Vertices, rep, thresholds, sgn, w1);
                direction = VectorMap<T>(w1.data(), w1.size() - 1);
                delta = w1(w1.size() - 1);
                if (delta <= 0)
                    break;
            }

            for (uint_fast64_t i = 0; i < direction.size(); ++i) {
                w_[i] = (double) (sgn(i) * direction(i));
            }
            this->VIhandler->valueIteration(w_);
            r_ = this->VIhandler->getResults();
            //r_.resize(nObjs);
            for (uint_fast64_t i = 0; i < nObjs; ++i) {
                vertex(i) = (T) r_[i];
            }
            Vertices.push_back(vertex);
            Directions.push_back(direction);

            Vector<T> wTemp = (sgn.array() * direction.array()).matrix();
            if (wTemp.dot(thresholds - vertex) > 0) {
                achievable = false;
                ++iter;
                break;
            }
            //std::cout << "weighted value: " << cudaVIHandler.getResults()[m]<<"\n";
            ++iter;
        }
        this->VIhandler->exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "Achievability Query terminates after " << iter << " iteration(s) \n";
        std::cout << "OUTPUT: " << std::boolalpha << achievable << "\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class AchievabilityQuery<double, int>;
}