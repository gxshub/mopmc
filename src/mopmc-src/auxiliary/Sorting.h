//
// Created by guoxin on 26/12/23.
//

#ifndef MOPMC_SORTING_H
#define MOPMC_SORTING_H

#include <Eigen/Dense>
#include <numeric>

namespace mopmc::optimization::auxiliary {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    enum SORTING_DIRECTION { ASCENT,
                             DECENT };

    template<typename V>
    class Sorting {
    public:
        static std::vector<size_t> argsort(const Vector<V> &vec, SORTING_DIRECTION direction = SORTING_DIRECTION::ASCENT);
    };

}// namespace mopmc::optimization::auxiliary

#endif//MOPMC_SORTING_H
