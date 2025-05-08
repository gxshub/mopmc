//
// Created by guoxin on 8/05/25.
//

#ifndef MOPMC_EXPORTER_H
#define MOPMC_EXPORTER_H

#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>


namespace mopmc::exporter {

    bool exportModel(std::string const &path_to_model,
                     std::string const &property_string,
                     std::string const &modelExportPathPrefix);

    template<typename M>
    bool writeTransitionMatrix(const M &model, const std::string &filename);

    bool writeSchedulerReturn(const std::vector<std::vector<int>> &schedulerCollection,
                              const std::vector<double> &schedulerDistribution,
                              const std::string &filename
    );

    bool writeStateMap(const std::map<uint64_t, std::string> &stateMap,
                       const std::string &filename);

}
#endif //MOPMC_EXPORTER_H
