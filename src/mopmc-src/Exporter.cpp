//
// Created by guoxin on 8/05/25.
//

#include "Exporter.h"
#include <Eigen/Dense>
#include <cstdio>
#include <ctime>
#include <filesystem>
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
                     std::string const &modelExportFolderPath) {
        // Check if the directory exists and create it if necessary
        // This prevents errors if the target folder doesn't exist
        std::error_code ec; // For error handling with create_directories
        if (!std::filesystem::create_directories(modelExportFolderPath, ec)) {
            // Check if creation failed AND it wasn't because the directory already exists
            if (ec && ec != std::make_error_code(std::errc::file_exists)) {
                std::cerr << "Error creating directory '" << modelExportFolderPath << "': " << ec.message() << "\n";
                return false; // Return false if directory creation failed
            }
        }

        storm::utility::setUp();
        storm::settings::initializeAll("storm-starter-project", "storm-starter-project");
        auto buildResult = mopmc::ModelBuilder<ModelType>::buildOnly(path_to_model, property_string, true);
        return mopmc::exporter::writeStateMap(buildResult, modelExportFolderPath) && \
            mopmc::exporter::writeTransitionMatrix(buildResult, modelExportFolderPath);
    };

    template <typename M>
    bool writeStateMap(const mopmc::ModelBuildResult<M> &modelBuildResult,
                       const std::string &folderPath) {
        const std::string fixedFileName = "states.txt";
        std::filesystem::path filePath = std::filesystem::path(folderPath) / fixedFileName;

        std::ofstream outfile(filePath);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << " for writing." << std::endl;
            return false; // Indicate failure
        }

        auto stateToId = modelBuildResult.stateLookup->stateToId;
        auto varInfo = modelBuildResult.stateLookup->varInfo;
        std::map<uint64_t, std::string> stateMap;
        for (auto pair: stateToId) {
            stateMap[pair.second] = storm::generator::toString(pair.first, varInfo);
        }

        auto model = modelBuildResult.model;
        auto transitionMatrix = model.getTransitionMatrix();

        uint64_t count = 1;
        outfile << "state_id\texpression\tnon-deterministic\n";
        for (const auto &pair: stateMap) {
            uint64_t s = pair.first;
            auto c = transitionMatrix.getRowGroupIndices()[s];
            auto c_next = transitionMatrix.getRowGroupIndices()[s + 1];
            std::string nd = c_next - c == 1 ? "n" : "y";
            outfile << pair.first << "\t" << pair.second << "\t" << nd << std::endl;
            ++count;
        }

        outfile.close();
        std::cout << "Successfully wrote state map to " << filePath << ", " << count << " lines in total\n";
        return true;
    }

    template<typename M>
    bool writeTransitionMatrix(const mopmc::ModelBuildResult<M> &modelBuildResult, const std::string &folderPath) {
        const std::string fixedFileName = "transition_matrix.txt";
        std::filesystem::path filePath = std::filesystem::path(folderPath) / fixedFileName;
        std::ofstream outfile(filePath);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << " for writing." << std::endl;
            return false; // Indicate failure
        }
        // write transition matrix
        outfile << "state_pre\t" << "action\t" << "state_post\t" << "prob\n";
        auto model = modelBuildResult.model;
        const storm::storage::SparseMatrix<typename M::ValueType> &transitionMatrix = model.getTransitionMatrix();
        Eigen::SparseMatrix<typename M::ValueType, Eigen::RowMajor> eigenTransitionMatrix =
                *storm::adapters::EigenAdapter::toEigenSparseMatrix(transitionMatrix);
        eigenTransitionMatrix.makeCompressed();

        uint64_t numStates = eigenTransitionMatrix.innerSize();
        assert(numStates == transitionMatrix.getRowGroupCount());
        assert(numStates == transitionMatrix.getRowGroupIndices().size() - 1);
        uint64_t count = 1;
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
        std::cout << "Successfully wrote transition matrix to " << filePath << ", " << count << " lines in total"
                  << std::endl;
        return true;
    };

    bool writeSchedulerReturn(const std::vector<std::vector<int>> &schedulerCollection,
                              const std::vector<double> &schedulerDistribution,
                              const std::string &folderPath
    ) {
        // Check if the directory exists and create it if necessary
        // This prevents errors if the target folder doesn't exist
        std::error_code ec; // For error handling with create_directories
        if (!std::filesystem::create_directories(folderPath, ec)) {
            // Check if creation failed AND it wasn't because the directory already exists
            if (ec && ec != std::make_error_code(std::errc::file_exists)) {
                std::cerr << "Error creating directory '" << folderPath << "': " << ec.message() << "\n";
                return false; // Return false if directory creation failed
            }
        }
        const std::string schDistFileName = "scheduler_distribution.txt";
        const std::string schedulersFileName = "schedulers.txt";
        std::filesystem::path schDistFilePath = std::filesystem::path(folderPath) / schDistFileName;
        std::filesystem::path schedulersFilePath = std::filesystem::path(folderPath) / schedulersFileName;
        std::ofstream outfile1(schDistFilePath);
        std::ofstream outfile2(schedulersFilePath);
        if (!outfile1.is_open()) {
            std::cerr << "Error: Could not open file " << schDistFilePath << " for writing." << std::endl;
            return false; // Indicate failure
        }
        if (!outfile2.is_open()) {
            std::cerr << "Error: Could not open file " << schedulersFilePath << " for writing." << std::endl;
            return false; // Indicate failure
        }

        // write scheduler distribution
        outfile1 << "scheduler_id\tprobability\n";
        for (size_t i = 0; i < schedulerDistribution.size(); ++i) {
            outfile1 << i << "\t" << schedulerDistribution[i] <<"\n";
        }
        // write schedulers
        int count = 0;
        outfile2 << "scheduler\tstate\taction\n";
        for (auto sch: schedulerCollection) {
            for (size_t i = 0; i < sch.size(); ++i) {
                outfile2 << count << "\t" << i << "\t" << sch[i] << "\n";
            }
            ++count;
        }
        outfile1.close();
        outfile2.close();
        std::cout << "Successfully wrote content to " << schDistFilePath << " and " << schedulersFilePath << std::endl;
        return true;
    }

}
