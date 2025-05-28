//
// Created by guoxin on 8/04/24.
//

#ifndef MOPMC_PRINTER_H
#define MOPMC_PRINTER_H

#include <Eigen/Dense>
#include <iostream>

namespace mopmc {
    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class [[maybe_unused]] Printer{
    public:
        [[maybe_unused]] static void printVector(const Vector<V> &vec) {
            printVector("", vec);
        }

        [[maybe_unused]] static void printVector(const std::vector<V> &vec) {
            printVector("", vec);
        }

        [[maybe_unused]] static void printVector(const std::string& str, const Vector<V> &vec) {
            if (!str.empty()) {
                std::cout << str << ": ";
            }
            bool firstElement = true;
            for (uint64_t i = 0; i < vec.size(); ++i) {
                if (firstElement) {
                    std::cout << "[" << vec(i);
                    firstElement = false;
                } else {
                    std::cout << " "  << vec(i);
                }
            }
            std::cout << "]\n";
        }

        [[maybe_unused]] static void printVector(const std::string& str, const std::vector<V> &vec) {
            if (!str.empty()) {
                std::cout << str << ": ";
            }
            bool firstElement = true;
            for (uint64_t i = 0; i < vec.size(); ++i) {
                if (firstElement) {
                    std::cout << "[" << vec[i];
                    firstElement = false;
                } else {
                    std::cout << " "  << vec[i];
                }
            }
            std::cout << "]\n";
        }
    };
}

#endif//MOPMC_PRINTER_H
