//
// Created by guoxin on 2/11/23.
//


#include "Runner.h"
#include "Transformation.h"
#include "convex-functions/EuclideanDistance.h"
#include "convex-functions/MSE.h"
#include "convex-functions/Variance.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"
#include "optimizers/FrankWolfe.h"
#include "optimizers/FrankWolfeOuterPolytope.h"
#include "optimizers/ProjectedGradientDescent.h"
#include "queries/AchievabilityQuery.h"
#include "queries/ConvexQuery.h"
#include <Eigen/Dense>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/utility/initialize.h>
#include <string>

namespace mopmc {

    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::models::sparse::Mdp<double>::ValueType ValueType;
    typedef storm::storage::sparse::state_type IndexType;

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    void printResult(const mopmc::queries::AchievabilityQuery<ValueType, int>& q);
    void printResult(const mopmc::queries::ConvexQuery<ValueType, int>& q);

    bool run(std::string const &path_to_model, std::string const &property_string, QueryOptions queryOptions) {
        assert(typeid(ValueType) == typeid(double));
        assert(typeid(IndexType) == typeid(uint64_t));

        // Init loggers
        storm::utility::setUp();
        storm::settings::initializeAll("storm-starter-project", "storm-starter-project");
        storm::Environment env;
        clock_t time0 = clock();
        auto preprocessedResult = mopmc::ModelBuilder<ModelType>::preprocess(path_to_model, property_string, env);
        clock_t time05 = clock();
        auto preparedModel = mopmc::ModelBuilder<ModelType>::build(preprocessedResult);
        clock_t time1 = clock();
        auto data = mopmc::Transformation<ModelType, ValueType, IndexType>::transform_i32_v2(preprocessedResult,
                                                                                             preparedModel);
        clock_t time2 = clock();

        mopmc::value_iteration::gpu::CudaValueIterationHandler<ValueType> cudaVIHandler(&data);
        switch (queryOptions.QUERY_TYPE) {
            case QueryOptions::ACHIEVABILITY: {
                mopmc::queries::AchievabilityQuery<ValueType, int> q(data, &cudaVIHandler);
                q.query();
                printResult(q);
                break;
            }
            case QueryOptions::CONVEX: {
                auto h = Eigen::Map<Vector<ValueType>>(data.thresholds.data(), data.thresholds.size());
                std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>> fn;
                switch (queryOptions.CONVEX_FUN) {
                    case QueryOptions::MSE: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::MSE<ValueType>(h, data.objectiveCount));
                        break;
                    }
                    case QueryOptions::EUD: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::EuclideanDistance<ValueType>(h));
                        break;
                    }
                    case QueryOptions::VAR: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::Variance<ValueType>(h.size()));
                        break;
                    }
                }
                /*
                std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>> optimizer;
                switch (queryOptions.INNER_OPTIMIZER) {
                    case QueryOptions::SIMPLEX_GD: {
                        optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                                new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                        mopmc::optimization::optimizers::FWOption::SIMPLEX_GD, &*fn));
                        break;
                    }
                    case QueryOptions::AWAY_STEP: {
                        optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                                new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                        mopmc::optimization::optimizers::FWOption::AWAY_STEP, &*fn));
                        break;
                    }
                    case QueryOptions::BLENDED: {
                        optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                                new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                        mopmc::optimization::optimizers::FWOption::BLENDED, &*fn));
                        break;
                    }
                    case QueryOptions::BLENDED_STEP_OPT: {
                        optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                                new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                        mopmc::optimization::optimizers::FWOption::BLENDED_STEP_OPT, &*fn));
                        break;
                    }
                    case QueryOptions::PGD: {
                        optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                                new mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType>(
                                        mopmc::optimization::optimizers::ProjectionType::UnitSimplex, &*fn));
                        break;
                    }
                }
                 */
                mopmc::optimization::optimizers::FrankWolfe<ValueType> innerOptimizer(mopmc::optimization::optimizers::FWOption::SIMPLEX_GD, &*fn);
                //mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType> projectedGD(&*fn);
                mopmc::optimization::optimizers::FrankWolfeOuterPolytope<ValueType> outerOptimizer(&*fn);
                mopmc::queries::ConvexQuery<ValueType, int> q(data, &*fn, &innerOptimizer, &outerOptimizer, &cudaVIHandler);
                q.query();
                printResult(q);
                break;
            }
        }
        clock_t time3 = clock();

        std::cout << "       TIME STATISTICS        \n";
        printf("Model building stage 1: %.3f seconds.\n", double(time05 - time0) / CLOCKS_PER_SEC);
        printf("Model building stage 2: %.3f seconds.\n", double(time1 - time05) / CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f seconds.\n", double(time2 - time1) / CLOCKS_PER_SEC);
        printf("Model checking: %.3f seconds.\n", double(time3 - time2) / CLOCKS_PER_SEC);
        return true;
    }

    void printResult(const mopmc::queries::AchievabilityQuery<ValueType, int>& q) {
        std::cout << "----------------------------------------------\n";
        std::cout << "Achievability Query terminates after " << q.getMainLoopIterationCount() << " iteration(s) \n";
        std::cout << "OUTPUT: " << std::boolalpha << q.getResult() << "\n";
        std::cout << "----------------------------------------------\n";
    };

    void printResult(const mopmc::queries::ConvexQuery<ValueType, int>& q) {
        std::cout << "----------------------------------------------\n"
                  << "CUDA CONVEX QUERY terminates after " << q.getMainLoopIterationCount() << " iteration(s)\n"
                  << "Estimated nearest point to threshold : [";
        for (int i = 0; i < q.getInnerOptimalPoint().size(); ++i) {
            std::cout << q.getInnerOptimalPoint()(i) << " ";
        }
        std::cout << "]\n"
                  << "Approximate distance (at inner point): " << q.getInnerOptimalValue()
                  << "\nApproximate distance (at outer point): " << q.getOuterOptimalValue()
                  << "\n----------------------------------------------\n";
    }
}// namespace mopmc