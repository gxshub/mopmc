//
// Created by guoxin on 18/12/23.
//

#include "MSE.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    MSE<V>::MSE(const Vector<V> &c, const uint64_t &n) : BaseConvexFunction<V>(c) {
        this->smooth = true;
    }

    template<typename V>
    MSE<V>::MSE(const uint64_t &n) : BaseConvexFunction<V>(n) {
        this->smooth = true;
    }

    template<typename V>
    V MSE<V>::value(const Vector<V> &x) {
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - this->parameters(i), 2);
        }
        return y / this->dimension;
    }

    template<typename V>
    Vector<V> MSE<V>::subgradient(const Vector<V> &x) {
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - this->parameters(i));
        }
        return y / this->dimension;
    }

    template<typename V>
    RMSE<V>::RMSE(const Vector<V> &c, const uint64_t &n) : BaseConvexFunction<V>(c) {
        this->smooth = true;
    }

    template<typename V>
    V RMSE<V>::value(const Vector<V> &x) {
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - this->parameters(i), 2);
        }
        return std::sqrt(y / this->dimension);
    }

    template<typename V>
    Vector<V> RMSE<V>::subgradient(const Vector<V> &x) {
        //todo
        V g = this->value(x);
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - this->parameters(i));
        }
        return y / this->dimension;
    }

    template class MSE<double>;
    template class RMSE<double>;
}// namespace mopmc::optimization::convex_functions