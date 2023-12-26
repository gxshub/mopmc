//
// Created by guoxin on 26/12/23.
//

#include "Sorting.h"

namespace mopmc::optimization::auxiliary {

    template<typename V>
    std::vector<size_t> Sorting<V>::argsort(const Vector<V> &vec, SORTING_DIRECTION direction) {
        std::vector<size_t> indices(vec.size());
        std::iota(indices.begin(), indices.end(), 0);
        switch (direction) {
            case ASCENT:
                std::sort(indices.begin(), indices.end(),
                          [&vec](int left, int right) -> bool {
                              // sort indices according to corresponding array element
                              return vec(left) < vec(right);
                          });
                break;
            case DECENT:
                std::sort(indices.begin(), indices.end(),
                          [&vec](int left, int right) -> bool {
                              // sort indices according to corresponding array element
                              return vec(left) > vec(right);
                          });
                break;
        }
        return indices;
    }

    template class Sorting<double>;
}// namespace mopmc::optimization::auxiliary