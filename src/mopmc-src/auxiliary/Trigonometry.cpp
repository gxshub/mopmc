//
// Created by guoxin on 27/12/23.
//

#include "Trigonometry.h"

namespace mopmc::optimization::auxiliary {

    template<typename V>
    V Trigonometry<V>::cosine(const Vector<V> &x, const Vector<V> &y, const V &dflt) {
        V a = x.template lpNorm<2>();
        V b = y.template lpNorm<2>();
        if (a == 0 || b == 0) {
            return dflt;
        } else {
            return x.dot(y) / (a * b);
        }
    }

    template class Trigonometry<double>;
}// namespace mopmc::optimization::auxiliary