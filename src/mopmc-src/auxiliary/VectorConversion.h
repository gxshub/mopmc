//
// Created by guoxin on 7/05/25.
//

#ifndef MOPMC_VECTORCONVERSION_H
#define MOPMC_VECTORCONVERSION_H


#include <Eigen/Dense>
#include <vector>
#include <iostream>


namespace mopmc::optimization::auxiliary{

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template <typename V>
    class VectorConversion {

    public:
        static void eigenToStdVector(const Vector<V> &eigenVec, std::vector<V> &stdVec){
            stdVec.resize(eigenVec.size());
            for (int64_t i = 0; i < stdVec.size(); ++i) {
                stdVec[i] = eigenVec(i);
            }
        };

        static Vector<V> stdToEigenVector(const std::vector<V> &stdVec, Vector<V> &eigenVec) {
            eigenVec.resize(stdVec.size());
            for (int64_t i = 0; i < eigenVec.size(); ++i) {
                eigenVec(i) = stdVec[i];
            }
        }


    };
}

#endif //MOPMC_VECTORCONVERSION_H
