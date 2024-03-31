//
// Created by guoxin on 30/03/24.
//

#ifndef MOPMC_CONSTRAINEDCONVEXQUERY_H
#define MOPMC_CONSTRAINEDCONVEXQUERY_H

#include "BaseQuery.h"
#include "mopmc-src/QueryData.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V, typename I>
    class ConstrainedConvexQuery : public BaseQuery<V, I> {
    public:
        ConstrainedConvexQuery(const mopmc::QueryData<V,I> &data,
                    mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *innerOptimizer,
                    mopmc::optimization::optimizers::BaseOptimizer<V> *outerOptimizer,
                    mopmc::value_iteration::BaseVIHandler<V> *valueIteration)
            : BaseQuery<V, I>(data, f, innerOptimizer, outerOptimizer, valueIteration) {
            innerPoint.resize(data.objectiveCount);
            outerPoint.resize(data.objectiveCount);
        };

        void query() override;

        [[nodiscard]] uint_fast64_t getMainLoopIterationCount() const {
            return iter;
        }
        const Vector<V> &getInnerOptimalPoint() const {
            return innerPoint;
        }
        const Vector<V> &getOuterOptimalPoint() const {
            return outerPoint;
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
        bool checkConstraint(const Vector<V> &point);
        uint_fast64_t iter{};
        Vector<V> innerPoint, outerPoint;
        std::vector<Vector<V>> Vertices, Points, Directions;
        bool feasible{true};
    };

}// namespace mompc::queries

#endif//MOPMC_CONSTRAINEDCONVEXQUERY_H
