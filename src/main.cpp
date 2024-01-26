
#include <iostream>
#include <boost/program_options.hpp>
#include "mopmc-src/Runner.h"
#include "mopmc-src/QueryOptions.h"

namespace po = boost::program_options;
using namespace std;

int main (int ac, char *av[]) {

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help,h", "produce help message")
                ("model,m", po::value<string>(), "model")
                ("prop,p", po::value<string>(), "multi-objective property")
                ("loss,l", po::value<string>()->default_value("mse"), "convex function")
                ("query,q", po::value<string>(), "query type (convex or achievability)")
                ;
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        string modelFile, propsFile;
        if (vm.count("prism") && vm.count("props")) {
            modelFile = vm["prism"].as<string>();
            propsFile = vm["props"].as<string>();
        } else {
            cout << "model and/or property not specified\n";
            return 1;
        }

        mopmc::QueryOptions queryOptions{};
        if (vm.count("query")) {
            const auto& s = vm["query"].as<string>();
            if (s == "achievability") {
                queryOptions.QUERY_TYPE = mopmc::QueryOptions::ACHIEVABILITY;
            } else if (s == "convex") {
                queryOptions.QUERY_TYPE = mopmc::QueryOptions::CONVEX;
            } else {
                cout << "not supported query type\n";
                return 1;
            }
        }
        if (vm.count("loss")) {
            const auto& s = vm["loss"].as<string>();
            if (s == "mse") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::MSE;
            } else if (s == "eud") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::EUD;
            } else if (s == "var") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::VAR;
            } else {
                cout << "not supported convex function\n";
                return 1;
            }
        }
        mopmc::run(modelFile, propsFile, queryOptions);
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
        return 1;
    }
    return 0;
}
