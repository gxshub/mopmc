//
// Created by guoxin on 8/05/25.
//

#include "Exporter.h"
#include "ModelBuilder.h"
#include <Eigen/Dense>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/utility/initialize.h>
#include <storm/api/storm.h>
#include <storm/adapters/EigenAdapter.h>
#include <string>

namespace mopmc::exporter {
    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::models::sparse::Mdp<double>::ValueType ValueType;
    typedef storm::storage::sparse::state_type IndexType;

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    bool exportModel(std::string const &path_to_model,
                     std::string const &property_string,
                     std::string const &modelExportPathPrefix) {
        storm::utility::setUp();
        storm::settings::initializeAll("storm-starter-project", "storm-starter-project");
        bool lookup = true;
        auto buildResult = mopmc::ModelBuilder<ModelType>::build(path_to_model, property_string, lookup);
        auto stateToId = buildResult.stateLookup->stateToId;
        auto varInfo = buildResult.stateLookup->varInfo;
        std::map<uint64_t, std::string> stateMap;
        for (auto pair: stateToId) {
            stateMap[pair.second] = storm::generator::toString(pair.first, varInfo);
        }
        std::string stateMapPath = modelExportPathPrefix + "_states.txt";
        std::string tranMatrixPath = modelExportPathPrefix + "_transition_matrix.txt";
        return mopmc::exporter::writeStateMap(stateMap, stateMapPath) && \
            mopmc::exporter::writeTransitionMatrix(buildResult.model, tranMatrixPath);
    };

    template<typename M>
    bool writeTransitionMatrix(const M &model, const std::string &filename) {
        // Open the file for writing. If the file exists, it will be overwritten.
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false; // Indicate failure
        }
        // write transition matrix
        outfile << "state (pre)\t" << "action\t" << "state (post)\t" << "prob\n";
        const storm::storage::SparseMatrix<typename M::ValueType> &transitionMatrix = model.getTransitionMatrix();
        Eigen::SparseMatrix<typename M::ValueType, Eigen::RowMajor> eigenTransitionMatrix =
                *storm::adapters::EigenAdapter::toEigenSparseMatrix(transitionMatrix);
        eigenTransitionMatrix.makeCompressed();

        uint64_t numStates = eigenTransitionMatrix.innerSize();
        assert(numStates == transitionMatrix.getRowGroupCount());
        assert(numStates == transitionMatrix.getRowGroupIndices().size() - 1);
        uint64_t count = 0;
        for (size_t s = 0; s < numStates; ++s) {
            auto offset = transitionMatrix.getRowGroupIndices()[s];
            auto offset_next = transitionMatrix.getRowGroupIndices()[s + 1];
            for (size_t a = 0; a < offset_next - offset; ++a) {
                auto start = eigenTransitionMatrix.outerIndexPtr()[a + offset];
                auto end = eigenTransitionMatrix.outerIndexPtr()[a + offset + 1];
                for (size_t j = start; j < end; ++j) {
                    auto c = eigenTransitionMatrix.innerIndexPtr()[j];
                    auto v = eigenTransitionMatrix.valuePtr()[j];
                    outfile << s << "\t"; //state
                    outfile << a << "\t"; //action
                    outfile << c << "\t"; //state
                    outfile << v << "\n"; //transition probability
                    ++count;
                }
            }
        };
        outfile.close();
        std::cout << "Successfully wrote transition matrix to " << filename << ", " << count << " lines in total"
                  << std::endl;
        return true;
    };

    bool writeSchedulerReturn(const std::vector<std::vector<int>> &schedulerCollection,
                              const std::vector<double> &schedulerDistribution,
                              const std::string &filename
    ) {
        // Open the file for writing. If the file exists, it will be overwritten.
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false; // Indicate failure
        }
        // write scheduler distribution
        outfile << "--distribution of schedulers--\n";
        bool firstElement = true;
        for (size_t i = 0; i < schedulerDistribution.size(); ++i) {
            outfile << i << ": " << schedulerDistribution[i];
            if (!firstElement) {
                outfile << ", ";
            }
            firstElement = false;
        }
        outfile << "\n";
        // write scheduler collection
        int count = 0;
        for (auto sch: schedulerCollection) {
            outfile << "--scheduler: " << count << "--\n";
            outfile << "state_id\taction_id\n";
            for (size_t i = 0; i < sch.size(); ++i) {
                outfile << i << "\t" << sch[i] << "\n";
            }
            ++count;
        }
        outfile.close();
        std::cout << "Successfully wrote vector to " << filename << std::endl;
        return true;
    }

    bool writeStateMap(const std::map<uint64_t, std::string> &stateMap,
                       const std::string &filename) {

        // Open the file for writing. If the file exists, it will be overwritten.
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false; // Indicate failure
        }
        uint64_t count = 0;
        for (const auto &pair: stateMap) {
            // pair.first is the key (long int)
            // pair.second is the value (std::string)

            // Write the key, a separator, and the value to the file, followed by a newline.
            outfile << pair.first << ": " << pair.second << std::endl;
            ++count;
        }

        outfile.close();
        std::cout << "Successfully wrote state map to " << filename << ", " << count << " lines in total\n";
        return true;
    }
}
