//
// Created by guoxin on 20/11/23.
//

#ifndef MOPMC_TRANSFORMATION_H
#define MOPMC_TRANSFORMATION_H

#include "QueryData.h"
#include "ModelBuilder.h"
#include "mopmc-src/_legacy/strom-wrappers/StormModelBuildingWrapper.h"
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/storage/SparseMatrix.h>
#include <string>
#include <storm/logic/MultiObjectiveFormula.h>

namespace mopmc {
    template<typename M, typename V, typename I>
    class Transformation {
    public:
        static mopmc::QueryData<V, int> transform(const ModelBuildResult<M> &buildResult);
        static mopmc::QueryData<V, int> transform(const ModelBuildAndProcessResult<M> &buildAndProcessResult);
    };
}// namespace mopmc

#endif//MOPMC_TRANSFORMATION_H
