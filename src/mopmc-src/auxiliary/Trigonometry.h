//
// Created by guoxin on 27/12/23.
//

#ifndef MOPMC_TRIGONOMETRY_H
#define MOPMC_TRIGONOMETRY_H

#include <Eigen/Dense>

namespace mopmc::optimization::auxiliary {


    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class Trigonometry {
    public:
        static V cosine(const Vector<V> &x, const Vector<V> &y, const V &dflt);
    };
}// namespace mopmc::optimization::auxiliary


#endif//MOPMC_TRIGONOMETRY_H
