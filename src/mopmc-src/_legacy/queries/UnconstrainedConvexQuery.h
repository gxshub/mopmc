//
// Created by guoxin on 16/01/24.
//

#ifndef MOPMC_UNCONSTRAINEDCONVEXQUERY_H
#define MOPMC_UNCONSTRAINEDCONVEXQUERY_H
#include "mopmc-src/queries/BaseQuery.h"
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
    class UnconstrainedConvexQuery : public BaseQuery<V, I> {
    public:
        UnconstrainedConvexQuery(const mopmc::QueryData<V, I> &data,
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
        uint_fast64_t iter{};
        Vector<V> innerPoint, outerPoint;
        std::vector<Vector<V>> Vertices, Directions;
        bool assertSeparation(const Vector<V> &point, const Vector<V> &direction);
    };
}// namespace mopmc::queries


#endif//MOPMC_UNCONSTRAINEDCONVEXQUERY_H
