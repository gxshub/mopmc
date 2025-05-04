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
    ModelBuildResult<ModelType> ModelBuilder<ModelType>::build(
            const std::string &path_to_model, const std::string &property_string) {
        //env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);

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

        std::cout << "number of states in original mdp: " << model->getNumberOfStates() << "\n";
        std::cout << "number of choices in original mdp: " << model->getNumberOfChoices() << "\n";
        std::cout << "number of transitions in original mdp: " << model->getTransitionMatrix().getEntryCount() <<"\n";
        return ModelBuildResult<ModelType>(*model, formula);
    }

    template<typename ModelType>
    ModelBuildAndProcessResult<ModelType> ModelBuilder<ModelType>::buildAndProcess(
            const std::string &path_to_model, const std::string &property_string,
            storm::Environment &env) {

        ModelBuildResult buildResult = ModelBuilder<ModelType>::build(path_to_model, property_string);
        ModelType model = buildResult.model;
        storm::logic::MultiObjectiveFormula formula  = buildResult.formula;
        env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);
        StormPreprocesReturn<ModelType> prepResult = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::preprocess(env, model, formula);
        mopmc::StormModelChecker processedResult = StormModelChecker<ModelType>(prepResult);

        std::cout << "number of states in pre-processed mdp: " << prepResult.preprocessedModel->getNumberOfStates() << "\n";
        std::cout << "number of choices in pre-processed mdp: " << prepResult.preprocessedModel->getNumberOfChoices() << "\n";
        std::cout << "number of transitions in pre-processed mdp: " << prepResult.preprocessedModel->getTransitionMatrix().getEntryCount() <<"\n";
        std::cout << "number of states in processed mdp: " << processedResult.getTransitionMatrix().getRowGroupCount() <<"\n";
        std::cout << "number of choices in processed mdp: " << processedResult.getTransitionMatrix().getRowCount() <<"\n";
        std::cout << "number of transitions in processed mdp: " << processedResult.getTransitionMatrix().getEntryCount() <<"\n";

        return ModelBuildAndProcessResult<ModelType> (model, formula, prepResult, processedResult);
    }

    template
    class ModelBuilder<storm::models::sparse::Mdp<double>>;

}