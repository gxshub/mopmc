//
// Created by guoxin on 2/11/23.
//

#include "Runner.h"
#include "Transformation.h"
#include "convex-functions/MSE.h"
#include "convex-functions/Variance.h"
#include "mopmc-src/solvers/CudaValueIteration.cuh"
#include "mopmc-src/solvers/ValueIteration.h"
#include "optimizers/FrankWolfeMethod.h"
#include "optimizers/MinimumNormPoint.h"
#include "optimizers/ProjectedGradient.h"
#include "queries/AchievabilityQuery.h"
#include "queries/ConvexQuery.h"
#include "mopmc-src/_legacy/queries/UnconstrainedConvexQuery.h"
#include "ModelBuilder.h"
#include "Exporter.h"
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

    bool run(std::string const &path_to_model,
             std::string const &property_string,
             QueryOptions options,
             std::string const &schedulerExportFolder,
             bool withProcessing) {
        assert(typeid(ValueType) == typeid(double));
        assert(typeid(IndexType) == typeid(uint64_t));

        storm::utility::setUp();
        storm::settings::initializeAll("storm-starter-project", "storm-starter-project");
        storm::Environment env;

        clock_t time0, time1, time2;
        QueryData<ValueType, int> data;

        time0 = clock();
        if (withProcessing) {
            auto buildAndProcessResult = mopmc::ModelBuilder<ModelType>::buildAndProcess(path_to_model,
                                                                                         property_string,
                                                                                         env);
            time1 = clock();
            data = mopmc::Transformation<ModelType, ValueType, IndexType>::transform(buildAndProcessResult);
        } else {
            auto buildResult = mopmc::ModelBuilder<ModelType>::buildOnly(path_to_model, property_string);
            time1 = clock();
            data = mopmc::Transformation<ModelType, ValueType, IndexType>::transform(buildResult);
            bool totalRewardFormulaOnly = true;
            for (const auto& f : buildResult.formula.getSubformulas()) {
                if (!f->asUnaryStateFormula().getSubformula().isTotalRewardFormula()){
                    totalRewardFormulaOnly = false;
                }
            }
            if (!totalRewardFormulaOnly) {
                std::cout << "! Model must be processed for query with non total rewards.\n";
                return false;
            }
        }
        time2 = clock();

        std::unique_ptr<mopmc::value_iteration::BaseVIHandler<ValueType>> vIHandler;
        std::unique_ptr<mopmc::queries::BaseQuery<ValueType, int>> q1;
        std::unique_ptr<mopmc::optimization::convex_functions::BaseConvexFunction<ValueType>> fn;
        auto h = Eigen::Map<Vector<ValueType>>(data.thresholds.data(), (int64_t) data.thresholds.size());

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
                q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType, int>>(
                        new mopmc::queries::AchievabilityQuery<ValueType, int>(data, &*vIHandler));
                break;
            }
            case QueryOptions::CONVEX: {
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
                    q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType, int>>(
                            new mopmc::queries::ConvexQuery<ValueType, int>(data, &*fn, &innerOptimizer,
                                                                            &outerOptimizer, &*vIHandler));
                } else {
                    // mopmc::optimization::optimizers::FrankWolfeMethod<ValueType> innerOptimizer(&*fn);
                    mopmc::optimization::optimizers::MinimumNormPoint<ValueType> innerOptimizer(&*fn);
                    //q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType,int>>(
                    //        new mopmc::queries::UnconstrainedConvexQuery<ValueType, int> (data, &*fn, &innerOptimizer, &outerOptimizer, &*vIHandler));
                    q1 = std::unique_ptr<mopmc::queries::BaseQuery<ValueType, int>>(
                            new mopmc::queries::ConvexQuery<ValueType, int>(data, &*fn, &innerOptimizer,
                                                                            &outerOptimizer, &*vIHandler, false));
                }
                break;
            }
        }
        q1->query();
        q1->printResult();

        const clock_t time3 = clock();
        const uint64_t mainLoopCount(q1->getMainLoopIterationCount());

        // schedulers export
        if (!schedulerExportFolder.empty()) {
            if (withProcessing) {
                std::cout << "! Schedulers export only implemented for query without model processing.\n";
            } else {
                mopmc::exporter::writeSchedulerReturn(q1->queryData.collectionOfSchedulers,
                                                      q1->queryData.schedulerDistribution,
                                                      schedulerExportFolder);
            }
        }

        std::cout << "       TIME STATISTICS        \n";
        printf("Model building stage: %.3f second(s).\n", double(time1 - time0) / CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f second(s).\n", double(time2 - time1) / CLOCKS_PER_SEC);
        printf("Model checking: %.3f second(s), %.3f per iteration.\n", double(time3 - time2) / CLOCKS_PER_SEC,
               double(time3 - time2) / CLOCKS_PER_SEC / static_cast<double>(mainLoopCount));
        printf("Total time: %.3f second(s).\n", double(time3 - time0) / CLOCKS_PER_SEC);
        return true;
    }

}// namespace mopmc