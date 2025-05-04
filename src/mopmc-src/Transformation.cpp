//
// Created by guoxin on 20/11/23.
//

#include "Transformation.h"
#include <Eigen/Sparse>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/adapters/EigenAdapter.h>
#include <storm/api/storm.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/solver/OptimizationDirection.h>
#include <storm/storage/prism/Program.h>

namespace mopmc {

    template<typename M, typename V, typename I>
    QueryData<V, int> Transformation<M, V, I>::transform(const ModelBuildResult<M> &buildResult){

        auto model = buildResult.model;
        auto formula = buildResult.formula;
        // size consistence check
        assert(formula.getReferencedRewardModels().size() == model.getRewardModels().size());

        mopmc::QueryData<V, int> data;
        data.transitionMatrix = *storm::adapters::EigenAdapter::toEigenSparseMatrix(model.getTransitionMatrix());
        data.transitionMatrix.makeCompressed();
        data.rowCount = model.getTransitionMatrix().getRowCount();
        data.colCount = model.getTransitionMatrix().getColumnCount();

        std::vector<uint64_t> rowGroupIndices1 = model.getTransitionMatrix().getRowGroupIndices();
        data.rowGroupIndices = std::vector<int>(rowGroupIndices1.begin(), rowGroupIndices1.end());
        data.row2RowGroupMapping.resize(data.rowCount);
        for (uint_fast64_t i = 0; i < data.rowGroupIndices.size() - 1; ++i) {
            size_t currInd = data.rowGroupIndices[i];
            size_t nextInd = data.rowGroupIndices[i + 1];
            for (uint64_t j = 0; j < nextInd - currInd; ++j)
                data.row2RowGroupMapping[currInd + j] = (int) i;
        }

        std::set<std::string> _allRewardModelNames = formula.getReferencedRewardModels();
        std::vector<std::string> allRewardModelNames(_allRewardModelNames.begin(), _allRewardModelNames.end());
        data.objectiveCount = allRewardModelNames.size();
        data.rewardVectors.resize(data.objectiveCount);
        data.thresholds.resize(data.objectiveCount);
        data.isProbabilisticObjective.resize(data.objectiveCount);
        data.isThresholdUpperBound.resize(data.objectiveCount);

        for (int i = 0; i < data.objectiveCount; ++i) {
            std::string rewModelName = allRewardModelNames[i];
            auto rewardModel = model.getRewardModel(rewModelName);
            if (!rewardModel.hasStateActionRewards()) {
                throw  std::runtime_error("To generate query input from an original model, "
                                          "only state-action reward structures are supported.");
            }
            data.rewardVectors[i] = rewardModel.getStateActionRewardVector();
            data.isProbabilisticObjective[i] = formula.getSubformula(i).isProbabilityOperatorFormula();
            data.thresholds[i] = formula.getSubformula(i).asOperatorFormula().template getThresholdAs<typename M::ValueType>();
            // When transforming from a model and a formula, >= is != Minimize.
            data.isThresholdUpperBound[i] = (formula.getSubformula(i).asOperatorFormula().getOptimalityType()
                                             != storm::solver::OptimizationDirection::Minimize);
        }
        data.flattenRewardVector.resize(data.objectiveCount * data.rowCount);
        for (uint_fast64_t i = 0; i < data.objectiveCount; ++i) {
            for (uint_fast64_t j = 0; j < data.rowCount; ++j) {
                data.flattenRewardVector[i * data.rowCount + j] = data.rewardVectors[i][j];
            }
        }

        data.scheduler.assign(data.colCount, static_cast<uint64_t>(0));
        for (auto s : model.getInitialStates()) {
            data.initialRow = (int) s;
        }

        return data;
    }

    template<typename M, typename V, typename I>
    QueryData<V, int> Transformation<M, V, I>::transform(const ModelBuildAndProcessResult<M> &buildAndProcessResult) {

        auto processedModel = buildAndProcessResult.processedModel;
        auto objectiveInformation = buildAndProcessResult.objectiveInformation;
        // size consistence check
        assert(objectiveInformation.objectives.size() == processedModel.getActionRewards().size());

        mopmc::QueryData<V, int> data;
        data.transitionMatrix = *storm::adapters::EigenAdapter::toEigenSparseMatrix(processedModel.getTransitionMatrix());
        data.transitionMatrix.makeCompressed();
        data.rowCount = processedModel.getTransitionMatrix().getRowCount();
        data.colCount = processedModel.getTransitionMatrix().getColumnCount();

        std::vector<uint64_t> rowGroupIndices1 = processedModel.getTransitionMatrix().getRowGroupIndices();
        data.rowGroupIndices = std::vector<int>(rowGroupIndices1.begin(), rowGroupIndices1.end());

        data.row2RowGroupMapping.resize(data.rowCount);
        for (uint_fast64_t i = 0; i < data.rowGroupIndices.size() - 1; ++i) {
            size_t currInd = data.rowGroupIndices[i];
            size_t nextInd = data.rowGroupIndices[i + 1];
            for (uint64_t j = 0; j < nextInd - currInd; ++j)
                data.row2RowGroupMapping[currInd + j] = (int) i;
        }

        data.objectiveCount = objectiveInformation.objectives.size();
        data.rewardVectors = processedModel.getActionRewards();
        data.flattenRewardVector.resize(data.objectiveCount * data.rowCount);
        for (uint_fast64_t i = 0; i < data.objectiveCount; ++i) {
            for (uint_fast64_t j = 0; j < data.rowCount; ++j) {
                data.flattenRewardVector[i * data.rowCount + j] = data.rewardVectors[i][j];
            }
        }
        data.thresholds.resize(data.objectiveCount);
        data.isProbabilisticObjective.resize(data.objectiveCount);
        data.isThresholdUpperBound.resize(data.objectiveCount);
        for (uint_fast64_t i = 0; i < data.objectiveCount; ++i) {
            data.thresholds[i] = objectiveInformation.objectives[i].formula->template getThresholdAs<V>();
            data.isProbabilisticObjective[i] = objectiveInformation.objectives[i].originalFormula->isProbabilityOperatorFormula();
            // When transforming from a processed model and objective information, >= is == Minimize
            data.isThresholdUpperBound[i] = (objectiveInformation.objectives[i].formula->getOptimalityType() == storm::solver::OptimizationDirection::Minimize);
        }

        data.scheduler.assign(data.colCount, static_cast<uint64_t>(0));
        data.initialRow = (int) processedModel.getInitialState();
        return data;
    }

    template class mopmc::Transformation<storm::models::sparse::Mdp<double>, double, uint64_t>;

}// namespace mopmc
