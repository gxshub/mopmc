//
// Created by guoxin on 2/11/23.
//


#include "Runner.h"
#include "Transformation.h"
#include "convex-functions/EuclideanDistance.h"
#include "convex-functions/MSE.h"
#include "convex-functions/Variance.h"
#include "mopmc-src/hybrid-computing/Problem.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"
#include "mopmc-src/storm-wrappers/StormModelCheckingWrapper.h"
#include "optimizers/FrankWolfe.h"
#include "optimizers/ProjectedGradientDescent.h"
#include "queries/AchievabilityQuery.h"
#include "queries/ConvexQuery.h"
#include "queries/ConvexQuery2.h"
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
        //mopmc::wrapper::StormModelCheckingWrapper<ModelType> stormModelCheckingWrapper(preprocessedResult);
        //stormModelCheckingWrapper.performMultiObjectiveModelChecking(env);
        auto preparedModel = mopmc::ModelBuilder<ModelType>::build(preprocessedResult);
        clock_t time1 = clock();
        auto data = mopmc::Transformation<ModelType, ValueType, IndexType>::transform_i32_v2(preprocessedResult,
                                                                                             preparedModel);
        clock_t time2 = clock();

        auto h = Eigen::Map<Vector<ValueType>>(data.thresholds.data(), data.thresholds.size());
        std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>> fn;
        switch (queryOptions.CONVEX_FUN) {
            case QueryOptions::MSE: {
                fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                        new mopmc::optimization::convex_functions::MSE<ValueType>(h, data.objectiveCount));
                break;
            }
            case QueryOptions::SE: {
                fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                        new mopmc::optimization::convex_functions::EuclideanDistance<ValueType>(h));
                break;
            }

            case QueryOptions::VAR: {
                fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                        new mopmc::optimization::convex_functions::Variance<ValueType>(h.size()));
                break;
            }
            case QueryOptions::SD: {
                fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                        new mopmc::optimization::convex_functions::StandDeviation<ValueType>(h.size()));
                break;
            }
        }
        std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>> optimizer;
        switch (queryOptions.PRIMARY_OPTIMIZER) {
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
            case QueryOptions::AWAY_STEP: {
                optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                        new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                mopmc::optimization::optimizers::FWOption::AWAY_STEP, &*fn));
                break;
            }
            case QueryOptions::LINOPT: {
                optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                        new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                mopmc::optimization::optimizers::FWOption::LINOPT, &*fn));
                break;
            }
            case QueryOptions::SIMPLEX_GD: {
                optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                        new mopmc::optimization::optimizers::FrankWolfe<ValueType>(
                                mopmc::optimization::optimizers::FWOption::SIMPLEX_GD, &*fn));
                break;
            }
            case QueryOptions::PGD: {
                optimizer = std::unique_ptr<mopmc::optimization::optimizers::BaseOptimizer<ValueType>>(
                        new mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType>(
                                mopmc::optimization::optimizers::ProjectionType::UnitSimplex, &*fn));
                break;
            }
        }
        mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType> projectedGD(&*fn);
        mopmc::value_iteration::gpu::CudaValueIterationHandler<ValueType> cudaVIHandler(&data);
        mopmc::queries::ConvexQuery<ValueType, int> q(data, &*fn, &*optimizer, &projectedGD, &cudaVIHandler);
        q.query();
        clock_t time3 = clock();

        std::cout << "       TIME STATISTICS        \n";
        printf("Model building stage 1: %.3f seconds.\n", double(time05 - time0) / CLOCKS_PER_SEC);
        printf("Model building stage 2: %.3f seconds.\n", double(time1 - time05) / CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f seconds.\n", double(time2 - time1) / CLOCKS_PER_SEC);
        printf("Model checking: %.3f seconds.\n", double(time3 - time2) / CLOCKS_PER_SEC);

        return true;
    }
}// namespace mopmc