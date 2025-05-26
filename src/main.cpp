
#include "mopmc-src/QueryOptions.h"
#include "mopmc-src/Runner.h"
#include "mopmc-src/Exporter.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;
using namespace std;

int main(int ac, char *av[]) {

    try {
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "produce help message")
                ("model,m", po::value<string>(), "model")
                ("prop,p", po::value<string>(), "multi-objective property")
                ("export,e", po::value<string>(), "folder to export model")
                ("query,q", po::value<string>(), "query type (convex or achievability)")
                ("loss,l", po::value<string>()->default_value("mse"), "convex function (mse or var)")
                ("constrained,c", po::value<string>()->default_value("y"), "constrained optimization (y or n)")
                ("value-iteration,v", po::value<string>()->default_value("gpu"), "value iteration method (gpu or standard)")
                ("export-scheduler,x", po::value<string>(), "folder to export schedulers");
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        string modelFile, propsFile;
        if (vm.count("model") && vm.count("prop")) {
            modelFile = vm["model"].as<string>();
            propsFile = vm["prop"].as<string>();
        } else {
            cout << "model and/or property not specified\n";
            return 1;
        }

        if (vm.count("export")) {
            string modelExportFolderPath = vm["export"].as<string>();
            mopmc::exporter::exportModel(modelFile, propsFile, modelExportFolderPath);
            return 0;
        }

        mopmc::QueryOptions queryOptions{};
        if (vm.count("query")) {
            const auto &s = vm["query"].as<string>();
            if (s == "achievability") {
                queryOptions.QUERY_TYPE = mopmc::QueryOptions::ACHIEVABILITY;
            } else if (s == "convex") {
                queryOptions.QUERY_TYPE = mopmc::QueryOptions::CONVEX;
            } else {
                cout << "unsupported query type\n";
                return 1;
            }
        }
        if (vm.count("loss")) {
            const auto &s = vm["loss"].as<string>();
            if (s == "mse") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::MSE;
            } else if (s == "var") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::VAR;
            } else {
                cout << "unsupported convex function\n";
                return 1;
            }
        }
        if (vm.count("value-iteration")) {
            const auto &s = vm["value-iteration"].as<string>();
            if (s == "gpu") {
                queryOptions.VI = mopmc::QueryOptions::CUDA_VI;
            } else if (s == "standard") {
                queryOptions.VI = mopmc::QueryOptions::STANDARD_VI;
            } else {
                cout << "unsupported value-iteration option\n";
                return 1;
            }
        }
        if (vm.count("constrained")) {
            const auto &s = vm["constrained"].as<string>();
            if (s == "y") {
                queryOptions.CONSTRAINED_OPT = mopmc::QueryOptions::CONSTRAINED;
            } else if (s == "n") {
                queryOptions.CONSTRAINED_OPT = mopmc::QueryOptions::UNCONSTRAINED;
            } else {
                cout << "incorrect constrained option\n";
                return 1;
            }
        }

        string schedulerExportFolderPath;
        if (vm.count("export-scheduler")) {
            schedulerExportFolderPath = vm["export-scheduler"].as<string>();
        }
        bool withModelProcessing = schedulerExportFolderPath.empty();

        mopmc::run(modelFile, propsFile, queryOptions, schedulerExportFolderPath, withModelProcessing);

    } catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        cerr << "Exception of unknown type!\n";
        return 1;
    }
    return 0;
}
