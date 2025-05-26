//
// Created by guoxin on 8/05/25.
//

#ifndef MOPMC_EXPORTER_H
#define MOPMC_EXPORTER_H

#include "ModelBuilder.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>


namespace mopmc::exporter {

    bool exportModel(std::string const &path_to_model,
                     std::string const &property_string,
                     std::string const &modelExportFolderPath);

    template <typename M>
    bool writeStateMap(const mopmc::ModelBuildResult<M> &modelBuildResult,
                       const std::string &folderPath);

    template<typename M>
    bool writeTransitionMatrix(const mopmc::ModelBuildResult<M> &modelBuildResult,
                               const std::string &folderPath);

    bool writeSchedulerReturn(const std::vector<std::vector<int>> &schedulerCollection,
                              const std::vector<double> &schedulerDistribution,
                              const std::string &folderPath
    );

}
#endif //MOPMC_EXPORTER_H
