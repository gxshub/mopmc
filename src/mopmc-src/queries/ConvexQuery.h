//
// Created by guoxin on 3/04/24.
//

#ifndef MOPMC_CONVEXQUERY_H
#define MOPMC_CONVEXQUERY_H
#include "BaseQuery.h"
#include "mopmc-src/QueryData.h"
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include <storm/storage/SparseMatrix.h>

namespace mopmc::queries {
    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V, typename I>
    class ConvexQuery : public BaseQuery<V, I> {
    public:
        ConvexQuery(const mopmc::QueryData<V, I> &data,
                    mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *innerOptimizer,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *outerOptimizer,
                    mopmc::value_iteration::BaseVIHandler<V> *valueIteration,
                    const bool withConstraint = true)
            : BaseQuery<V, I>(data, f, innerOptimizer, outerOptimizer, valueIteration) {
            innerPoint.resize(data.objectiveCount);
            outerPoint.resize(data.objectiveCount);
            hasConstraint = withConstraint;
            if (withConstraint)
                constraintsToHalfspaces();
        };
        void query() override;

        [[nodiscard]] uint64_t getMainLoopIterationCount() override {
            return this->iter;
        }
        const Vector<V> &getInnerOptimalPoint() const {
            return this->innerPoint;
        }
        const Vector<V> &getOuterOptimalPoint() const {
            return this->outerPoint;
        }
        V getInnerOptimalValue() const {
            return this->fn->value(innerPoint);
        }
        V getOuterOptimalValue() const {
            return this->fn->value(outerPoint);
        }

        void printResult() override;

    private:
        void constraintsToHalfspaces();
        bool checkConstraintSatisfaction(const Vector<V> &point);
        bool hasConstraint{true};
        uint_fast64_t iter{};
        Vector<V> innerPoint, outerPoint;
        std::vector<Vector<V>> Vertices, BoundaryPoints, Directions;
    };

}// namespace mopmc::queries


#endif//MOPMC_CONVEXQUERY_H
