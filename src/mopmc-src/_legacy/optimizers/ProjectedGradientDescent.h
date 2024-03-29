//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_PROJECTEDGRADIENTDESCENT_H
#define MOPMC_PROJECTEDGRADIENTDESCENT_H

#include "mopmc-src/auxiliary/Sorting.h"
#include "mopmc-src/convex-functions/BaseConvexFunction.h"
#include "mopmc-src/optimizers/BaseOptimizer.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace mopmc::optimization::optimizers {

    enum ProjectionType {
        NearestHyperplane,
        UnitSimplex
    };

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class ProjectedGradientDescent : public BaseOptimizer<V> {
    public:
        explicit ProjectedGradientDescent(convex_functions::BaseConvexFunction<V> *f) : BaseOptimizer<V>(f){};

        ProjectedGradientDescent(ProjectionType type, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : projectionType(type), BaseOptimizer<V>(f){};

        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Weights) override;

        ProjectionType projectionType{};

    private:
        Vector<V> minimizeByAdaptiveStepSize(const Vector<V> &point,
                                             const std::vector<Vector<V>> &Vertices,
                                             const std::vector<Vector<V>> &Directions);

        Vector<V> minimizeByFixedStepSize(const Vector<V> &point,
                                          const std::vector<Vector<V>> &Vertices,
                                          const std::vector<Vector<V>> &Directions);

        Vector<V> projectToHalfSpaces(const Vector<V> &point,
                                      const std::vector<Vector<V>> &Vertices,
                                      const std::vector<Vector<V>> &Directions);

        Vector<V> dykstrasProjection(const Vector<V> &point,
                                  const std::vector<Vector<V>> &Vertices,
                                  const std::vector<Vector<V>> &Directions);

        Vector<V> projectFromPointToHyperPlane(const Vector<V> &point, const Vector<V> &vertex, const Vector<V> &direction);

        Vector<V> intersectLineWithHalfSpaces(const Vector<V> &point1,
                                              const Vector<V> &point2,
                                              const std::vector<Vector<V>> &Vertices,
                                              const std::vector<Vector<V>> &Directions);

        Vector<V> argminUnitSimplexProjection(Vector<V> &weightVector,
                                              const std::vector<Vector<V>> &Points);

        Vector<V> projectToUnitSimplex(Vector<V> &x);

        Vector<V> optimalPoint;
        Vector<V> alpha;
    };
}// namespace mopmc::optimization::optimizers


#endif//MOPMC_PROJECTEDGRADIENTDESCENT_H
