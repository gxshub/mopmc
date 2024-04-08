//
// Created by guoxin on 2/11/23.
//

#include "Runner.h"
#include "Transformation.h"
#include "convex-functions/MSE.h"
#include "convex-functions/Variance.h"
#include "mopmc-src/solvers/CudaValueIteration.cuh"
#include "mopmc-src/solvers/ValueIteration.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"
#include "optimizers/FrankWolfeMethod.h"
#include "optimizers/MinimumNormPoint.h"
#include "optimizers/ProjectedGradient.h"
#include "queries/AchievabilityQuery.h"
#include "queries/ConvexQuery.h"
#include "queries/UnconstrainedConvexQuery.h"
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

    bool run(std::string const &path_to_model, std::string const &property_string, QueryOptions options) {
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
        std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>> q1;
        switch (options.VI) {
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
        switch (options.QUERY_TYPE) {
            case QueryOptions::ACHIEVABILITY: {
                q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>> (
                        new  mopmc::queries::AchievabilityQuery<ValueType, int>(data, &*vIHandler)
                        );
                q1->query();
                q1->printResult();
                break;
            }
            case QueryOptions::CONVEX: {
                auto h = Eigen::Map<Vector<ValueType>>(data.thresholds.data(), (int64_t) data.thresholds.size());
                std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>> fn;
                switch (options.CONVEX_FUN) {
                    case QueryOptions::MSE: {
                        if (options.CONSTRAINED_OPT == QueryOptions::CONSTRAINED) {
                            fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                    new mopmc::optimization::convex_functions::MSE<ValueType>(data.objectiveCount));
                        } else {
                            fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                              new mopmc::optimization::convex_functions::MSE<ValueType>(h, data.objectiveCount));
                            //fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                            //        new mopmc::optimization::convex_functions::MSE<ValueType>(data.objectiveCount));
                        }
                        break;
                    }
                    case QueryOptions::VAR: {
                        fn = std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>>(
                                new mopmc::optimization::convex_functions::Variance<ValueType>(h.size()));
                        break;
                    }
                }
                mopmc::optimization::optimizers::ProjectedGradient<ValueType> outerOptimizer(&*fn);
                if (options.CONSTRAINED_OPT == QueryOptions::CONSTRAINED) {
                    mopmc::optimization::optimizers::MinimumNormPoint<ValueType> innerOptimizer(&*fn);
                    q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>>(
                            new mopmc::queries::ConvexQuery<ValueType, int>(data, &*fn, &innerOptimizer, &outerOptimizer, &*vIHandler));
                } else {
                   // mopmc::optimization::optimizers::FrankWolfeMethod<ValueType> innerOptimizer(&*fn);
                    mopmc::optimization::optimizers::MinimumNormPoint<ValueType> innerOptimizer(&*fn);
                    //q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>>(
                    //        new mopmc::queries::UnconstrainedConvexQuery<ValueType, int> (data, &*fn, &innerOptimizer, &outerOptimizer, &*vIHandler));
                    q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>>(
                            new mopmc::queries::ConvexQuery<ValueType, int>(data, &*fn, &innerOptimizer, &outerOptimizer, &*vIHandler, false));
                }
                q1->query();
                q1->printResult();
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