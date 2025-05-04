//
// Created by guoxin on 17/11/23.
//

#include "StormModelBuildingWrapper.h"
#include <storm/generator/VariableInformation.h>
#include <Eigen/Sparse>
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
#include <storm/api/export.h>
#include <string>

namespace mopmc {

    //typedef storm::models::sparse::Mdp<double> ModelType;
    //typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    //typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;

    template<typename M>
    typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType
    MyModelBuilder<M>::preprocess(const std::string &path_to_model, const std::string &property_string,
                                  storm::Environment &env) {

        env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);

        //auto program = storm::parser::PrismParser::parse(path_to_model);
        storm::prism::Program program = storm::api::parseProgram(path_to_model);
        program = storm::utility::prism::preprocess(program, "");
        // Then parse the properties, passing the program to give context to some potential variables.
        auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
        // Translate properties into the more low-level formulae.
        auto formulas = storm::api::extractFormulasFromProperties(properties);

        //std::shared_ptr<storm::models::sparse::Mdp<typename M::ValueType>> mdp =
        //        storm::api::buildSparseModel<typename M::ValueType>(program, formulas)->template as<M>();

        storm::builder::BuilderOptions options(formulas, program);
        std::cout << "getInitialStatesExpression: " << program.getInitialStatesExpression().toString() << "\n";
        //auto varInfo = storm::generator::VariableInformation(program, options.getReservedBitsForUnboundedVariables(), options.isAddOutOfBoundsStateSet());


        std::shared_ptr<storm::generator::NextStateGenerator<typename M::ValueType, uint32_t>> generator;
        generator = std::make_shared<storm::generator::PrismNextStateGenerator<typename M::ValueType, uint32_t>>(
                static_cast<storm::storage::SymbolicModelDescription>(program).asPrismProgram(), options);
        storm::builder::ExplicitModelBuilder<typename M::ValueType> builder =
                storm::builder::ExplicitModelBuilder<typename M::ValueType>(generator);
        std::shared_ptr<storm::models::sparse::Mdp<typename M::ValueType>> mdp = builder.build()->template as<M>();

        auto varInfo = generator->getVariableInformation();
        storm::storage::BitVector initialStates = mdp->getInitialStates();
        std::cout << "initial state id (mdp->getInitialStates()): ";
        for (auto x: initialStates) {
            std::cout << x;
        }
        std::cout << "\n";
        auto initialStates1 = builder.stateStorage.initialStateIndices;
        std::cout << "initial state id (stateStorage.initialStateIndices):";
        for (auto num: initialStates1) {
            std::cout << num;
        }
        std::cout << "\n";
        /*
        if (mdp->hasLabel("done")) {
            auto labelStates = mdp->getStates("done");
            auto labelStates = mdp->getStates("done");
            std::cout << "label states (done): [";
            for (auto x : labelStates) {
                std::cout << x <<", ";
            }
            std::cout << "]\n";

        }
         */

        std::cout << "number of states in original mdp: " << mdp->getNumberOfStates() << "\n";
        std::cout << "number of choices in original mdp: " << mdp->getNumberOfChoices() << "\n";
        std::cout << "number of transitions in original mdp: " << mdp->getTransitionMatrix().getEntryCount() << "\n";

        const storm::logic::MultiObjectiveFormula formula = formulas[0]->asMultiObjectiveFormula();
        typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType prepResult =
                storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::preprocess(env,
                                                                                                                    *mdp,
                                                                                                                    formula);


        /*
        std::set<std::string> _rewardModelNames = formula.getReferencedRewardModels();
        std::vector<std::string> rewardModelNames(_rewardModelNames.begin(), _rewardModelNames.end());
        uint32_t numObjs = rewardModelNames.size();
        for (int i = 0; i < rewardModelNames.size(); ++i) {
            std::string rewModelName = rewardModelNames[i];
            std::cout << "reward model name from formula: " << rewModelName << "\n";
            std::cout << "length of reward vector: " << mdp->getRewardModel(rewModelName).getStateActionRewardVector().size() << "\n";
            auto h = formula.getSubformula(i).asOperatorFormula().template getThresholdAs<typename M::ValueType>();
            std::cout << "objective threshold: " << h << "\n";
        }
         */

        /*
        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<M>::analyze(prepResult);
        std::string s1 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::AllFinite ? "yes" : "no";
        std::string s2 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::ExistsParetoFinite ? "yes" : "no";
        std::cout << "[!] The expected reward is finite for all objectives and all schedulers: " << s1 << std::endl;
        std::cout << "[!] There is a Pareto optimal scheduler yielding finite rewards for all objectives: " << s2 << std::endl;
         */

        //std::ostream &outputStream = std::cout;
        //prepResult.preprocessedModel->printModelInformationToStream(outputStream);


        //Objectives must be total rewards
        if (!prepResult.containsOnlyTotalRewardFormulas()) {
            throw std::runtime_error("This framework handles total rewards only.");
        }
        //Confine the property (syntax) to achievability query
        // We will convert it to a convex query.
        if (prepResult.queryType !=
            storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<M>::QueryType::Achievability) {
            throw std::runtime_error("The input property should be achievability query type.");
        }

        //storm::storage::BitVector initialStates1 = prepResult.preprocessedModel->getInitialStates();
        //std::cout << "initial state (expression) (preprocessedModel): " << storm::generator::toString(initialStates1, varInfo) << "\n";
        //std::cout << "initial state (index) (preprocessedModel): " << builder.stateStorage.stateToId.getValue(initialStates1) << "\n";

        /*
        storm::generator::NextStateGeneratorOptions generatorOptions;
        generatorOptions.setBuildAllLabels();
        auto lookup = builder.exportExplicitStateLookup();
        auto xvar = program.getModules()[0].getIntegerVariable("x").getExpressionVariable();
        auto yvar = program.getModules()[0].getIntegerVariable("y").getExpressionVariable();
        auto zvar = program.getModules()[0].getIntegerVariable("z").getExpressionVariable();
        auto& manager = program.getManager();
        auto stateId0 = lookup.lookup({{xvar, manager.integer(0)}, {yvar, manager.integer(0)}, {zvar, manager.integer(0)}});
        auto stateId1 = lookup.lookup({{xvar, manager.integer(128)}, {yvar, manager.integer(0)}, {zvar, manager.integer(0)}});
        auto stateId2 = lookup.lookup({{xvar, manager.integer(0)}, {yvar, manager.integer(0)}, {zvar, manager.integer(1)}});
        auto stateId3 = lookup.lookup({{xvar, manager.integer(199)}, {yvar, manager.integer(99)}, {zvar, manager.integer(1)}});
        std::cout << "state lookup: " << stateId0 << "\n";
        std::cout << "state lookup: " << stateId1 << "\n";
        std::cout << "state lookup: " << stateId2 << "\n";
        std::cout << "state lookup: " << stateId3 << "\n";
        auto cs = storm::generator::createCompressedState(varInfo, {{xvar, manager.integer(128)},

        std::cout << "initial state (cs): " << storm::generator::toString(cs, varInfo) << "\n";
                                                        {yvar, manager.integer(0)}, {zvar, manager.integer(0)}}, true);
        */
        std::cout << "number of states in pre-processed mdp: " << prepResult.preprocessedModel->getNumberOfStates()
                  << "\n";
        std::cout << "number of choices in pre-processed mdp: " << prepResult.preprocessedModel->getNumberOfChoices()
                  << "\n";
        std::cout << "number of transitions in pre-processed mdp: "
                  << prepResult.preprocessedModel->getTransitionMatrix().getEntryCount() << "\n";


        return prepResult;
    }

    template<typename M>
    mopmc::MyModelBuilder<M> MyModelBuilder<M>::build(
            typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &preliminaryData) {

        MyModelBuilder<M> prepModel(preliminaryData);

        std::cout << "number of states in processed mdp: " << prepModel.getTransitionMatrix().getRowGroupCount()
                  << "\n";
        std::cout << "number of choices in processed mdp: " << prepModel.getTransitionMatrix().getRowCount() << "\n";
        std::cout << "number of transitions in processed mdp: " << prepModel.getTransitionMatrix().getEntryCount()
                  << "\n";
        return prepModel;
    }


    template
    class MyModelBuilder<storm::models::sparse::Mdp<double>>;
}// namespace mopmc
