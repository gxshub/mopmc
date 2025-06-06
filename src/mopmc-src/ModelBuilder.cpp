//
// Created by guoxin on 2/05/25.
//

#include "ModelBuilder.h"

//#include <Eigen/Sparse>
#include <iostream>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/adapters/EigenAdapter.h>
#include <storm/api/storm.h>
#include <storm/environment/Environment.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/prism/Program.h>
#include <string>

namespace mopmc {

    template<typename ModelType>
    ModelBuildResult<ModelType> ModelBuilder<ModelType>::buildOnly(
            const std::string &path_to_model, const std::string &property_string, const bool lookup) {

        //auto program = storm::parser::PrismParser::parse(path_to_model);
        storm::prism::Program program = storm::api::parseProgram(path_to_model);
        program = storm::utility::prism::preprocess(program, "");
        //std::cout << "Model Type: " << program.getModelType() << std::endl;
        // Then parse the properties, passing the program to give context to some potential variables.
        auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
        // Translate properties into the more low-level formulae.
        auto formulas = storm::api::extractFormulasFromProperties(properties);

        storm::builder::BuilderOptions options(formulas, program);
        std::shared_ptr<storm::generator::NextStateGenerator<typename ModelType::ValueType, uint32_t>> generator;
        generator = std::make_shared<storm::generator::PrismNextStateGenerator<typename ModelType::ValueType, uint32_t>>(
                static_cast<storm::storage::SymbolicModelDescription>(program).asPrismProgram(), options);
        storm::builder::ExplicitModelBuilder<typename ModelType::ValueType> builder =
                storm::builder::ExplicitModelBuilder<typename ModelType::ValueType>(generator);

        std::shared_ptr<ModelType> model = builder.build()->template as<ModelType>();
        storm::logic::MultiObjectiveFormula formula = formulas[0]->asMultiObjectiveFormula();

        std::cout << "--Model Building--\n";
        std::cout << "States:\t" << model->getNumberOfStates() << "\n";
        std::cout << "Choices:\t" << model->getNumberOfChoices() << "\n";
        std::cout << "Transitions:\t" << model->getTransitionMatrix().getEntryCount() <<"\n";

        storm::builder::ExplicitStateLookup<uint32_t> stateLookup = builder.exportExplicitStateLookup();
        auto stateLookupPtr = std::make_shared<storm::builder::ExplicitStateLookup<uint32_t>>(stateLookup);

        if (lookup) {
            return ModelBuildResult<ModelType>(*model, formula, stateLookupPtr);
        } else {
            return ModelBuildResult<ModelType>(*model, formula);
        }
    }

    template<typename ModelType>
    ModelBuildAndProcessResult<ModelType> ModelBuilder<ModelType>::buildAndProcess(
            const std::string &path_to_model, const std::string &property_string,
            storm::Environment &env) {

        ModelBuildResult buildResult = ModelBuilder<ModelType>::buildOnly(path_to_model, property_string);
        ModelType model = buildResult.model;
        storm::logic::MultiObjectiveFormula formula  = buildResult.formula;
        env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);
        StormPreprocesReturn<ModelType> prepResult = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::preprocess(env, model, formula);
        /*
        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<M>::analyze(prepResult);
        std::string s1 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::AllFinite ? "yes" : "no";
        std::string s2 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::ExistsParetoFinite ? "yes" : "no";
        std::cout << "[!] The expected reward is finite for all objectives and all schedulers: " << s1 << std::endl;
        std::cout << "[!] There is a Pareto optimal scheduler yielding finite rewards for all objectives: " << s2 << std::endl;
         */
        mopmc::StormModelChecker processedResult = StormModelChecker<ModelType>(prepResult);

        /* The following shows the model size change during internal model processing.
         * Reveal them if needed */
        //std::cout << "--Pre-processed Model (internal)--\n";
        //std::cout << "*States:\t" << prepResult.preprocessedModel->getNumberOfStates() << "\n";
        //std::cout << "*Choices:\t" << prepResult.preprocessedModel->getNumberOfChoices() << "\n";
        //std::cout << "*Transitions:\t" << prepResult.preprocessedModel->getTransitionMatrix().getEntryCount() <<"\n";
        //std::cout << "--Processed Model (internal)\n";
        //std::cout << "*States:\t" << processedResult.getTransitionMatrix().getRowGroupCount() <<"\n";
        //std::cout << "*Choices:\t" << processedResult.getTransitionMatrix().getRowCount() <<"\n";
        //std::cout << "*Transitions:\t" << processedResult.getTransitionMatrix().getEntryCount() <<"\n";

        return ModelBuildAndProcessResult<ModelType> (model, formula, prepResult, processedResult);
    }

    template
    class ModelBuilder<storm::models::sparse::Mdp<double>>;

}