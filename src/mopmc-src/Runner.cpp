//
// Created by guoxin on 2/11/23.
//

#include "Runner.h"
#include "Transformation.h"
#include "convex-functions/MSE.h"
#include "convex-functions/Variance.h"
#include "mopmc-src/_legacy/convex-functions/EuclideanDistance.h"
#include "mopmc-src/solvers/CudaValueIteration.cuh"
#include "mopmc-src/solvers/ValueIteration.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"
#include "optimizers/FrankWolfeInnerOptimizer.h"
#include "optimizers/ProjectedGradient.h"
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

        std::unique_ptr<mopmc::value_iteration::BaseVIHandler<ValueType>> vIHandler;
        switch (queryOptions.VI) {
            case QueryOptions::CUDA_VI: {
                vIHandler = std::unique_ptr<mopmc::value_iteration::BaseVIHandler<ValueType>>(
                        new mopmc::value_iteration::gpu::CudaValueIterationHandler<ValueType>(&data));
                break;
            }
            case QueryOptions::STANDARD_VI: {
                vIHandler = std::unique_ptr<mopmc::value_iteration::BaseVIHandler<ValueType>>(
                        new mopmc::value_iteration::ValueIterationHandler(&data));
                break;
            }
        }

        mopmc::value_iteration::gpu::CudaValueIterationHandler<ValueType> cudaVIHandler(&data);
        mopmc::value_iteration::ValueIterationHandler<ValueType> valueIterationHandler(&data);
        switch (queryOptions.QUERY_TYPE) {
            case QueryOptions::ACHIEVABILITY: {
                mopmc::queries::AchievabilityQuery<ValueType, int> q(data, &*vIHandler);
                q.query();
                q.printResult();
                break;
            }
            case QueryOptions::CONVEX: {
                auto h = Eigen::Map<Vector<ValueType>>(data.thresholds.data(), (int64_t) data.thresholds.size());
                std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>> fn;
                switch (queryOptions.CONVEX_FUN) {
                    case QueryOptions::MSE: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::MSE<ValueType>(h, data.objectiveCount));
                        break;
                    }
                    case QueryOptions::VAR: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::Variance<ValueType>(h.size()));
                        break;
                    }
                }
                mopmc::optimization::optimizers::FrankWolfeInnerOptimizer<ValueType> innerOptimizer(&*fn);
                mopmc::optimization::optimizers::ProjectedGradient<ValueType> outerOptimizer(&*fn);
                mopmc::queries::ConvexQuery<ValueType, int> q(data, &*fn, &innerOptimizer, &outerOptimizer, &*vIHandler);
                q.query();
                q.printResult();
                break;
            }
        }
        clock_t time3 = clock();

        std::cout << "       TIME STATISTICS        \n";
        printf("Model building stage 1: %.3f second(s).\n", double(time05 - time0) / CLOCKS_PER_SEC);
        printf("Model building stage 2: %.3f second(s).\n", double(time1 - time05) / CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f second(s).\n", double(time2 - time1) / CLOCKS_PER_SEC);
        printf("Model checking: %.3f second(s).\n", double(time3 - time2) / CLOCKS_PER_SEC);
        printf("Total time: %.3f second(s).\n", double(time3 - time0) / CLOCKS_PER_SEC);
        return true;
    }
}// namespace mopmc