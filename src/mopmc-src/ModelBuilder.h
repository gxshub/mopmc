//
// Created by guoxin on 2/05/25.
//

#ifndef MOPMC_MODELBUILDER_H
#define MOPMC_MODELBUILDER_H

#include <string>
#include <utility>
#include <storm/logic/MultiObjectiveFormula.h>

#include <storm/environment/Environment.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>

namespace mopmc {

    template<typename ModelType>
    using StormPreprocesReturn = typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType;

    template<typename ModelType>
    class StormModelChecker;

    template<typename ModelType>
    struct ModelBuildResult {

        ModelType model;
        storm::logic::MultiObjectiveFormula formula;

        ModelBuildResult() = default;
        ModelBuildResult(const ModelType &model,
                         const storm::logic::MultiObjectiveFormula &formula) :
                       model(model), formula(formula) {}
    };

    template<typename ModelType>
    struct ModelBuildAndProcessResult : public ModelBuildResult<ModelType> {

        StormModelChecker<ModelType> processedModel;
        StormPreprocesReturn<ModelType> objectiveInformation;

        ModelBuildAndProcessResult() = default;
        ModelBuildAndProcessResult(const ModelType &model,
                                   const storm::logic::MultiObjectiveFormula &formula,
                                   const StormPreprocesReturn<ModelType> preprocessedResult,
                                   const StormModelChecker<ModelType> &processedResult) :
                ModelBuildResult<ModelType>(model, formula),
                processedModel(processedResult),
                objectiveInformation(preprocessedResult) {}


    };


    template<typename ModelType>
    class ModelBuilder {
    public:
        explicit ModelBuilder() = default;

        static ModelBuildResult<ModelType> build(
                const std::string &path_to_model,
                const std::string &property_string);

        static ModelBuildAndProcessResult<ModelType> buildAndProcess(
                const std::string &path_to_model,
                const std::string &property_string,
                storm::Environment &env);
    };

    template<typename ModelType>
    class StormModelChecker : public storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType> {
    public:
        explicit StormModelChecker(StormPreprocesReturn<ModelType> returnType) :
                storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType>(returnType){};
        storm::storage::SparseMatrix<typename ModelType::ValueType> getTransitionMatrix() const {
            return this->transitionMatrix;
        }

        std::vector<std::vector<typename ModelType::ValueType>> getActionRewards() const{
            return this->actionRewards;
        }

        [[nodiscard]] uint64_t getInitialState() const {
            return this->initialState;
        }
    };


}

#endif //MOPMC_MODELBUILDER_H
