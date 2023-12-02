//
// Created by thomas on 22/10/23.
//

#ifndef MOPMC_LOOPER_H
#define MOPMC_LOOPER_H

#include <thread>
#include <atomic>
#include <memory>
#include <functional>
#include <stdexcept>
#include <queue>
#include <mutex>
#include <boost/optional.hpp>
#include <storm/storage/SparseMatrix.h>
#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include "Problem.h"
#include "Utilities.h"
// storm
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
namespace hybrid {
using namespace mopmc;

/*!
 * Loopers are objects which contain or are attached to a thread with a conditional infinite loop.
 * This loop runs as long as the abort-criteria is unmet. Within this loop, arbitrary actions can be
 * performed
 *
 * Implements start, run stop
 *
 * Define a CLooper class, which contains std::thread-member and a run-method which will create
 * the thread invoking runFunc - the second method - implementing an effective thread operation.
 *
 * Stopping the loop:
 *  In order to stop the looper, add an abort criteria to the infinite loop -mAbortRequested
 *  of type std::atomic<bool> which is checked against each iteration.
 *
 *  Add in a private method abortAndJoin() which will set the mAbortRequested flag to true -
 *  invokes join() on the thread and waits until the looper-function has been exited and the worker
 *  thread was joined.
 *
 *  The destructor will invoke abortAndJoin() iun the case the looper goes out of scope.
 *
 * Tasks:
 *  In the context of loopers, tasks are executable portions of code sharing a common signature
 *  i.e. one or more Tasks, which can be fetched from an internal collection (e.g. FIFO queue)
 *  and can be executed by the worker thread
 *
 * Dispatchers:
 *  Tasks are pushed to the queue with dispatching.
 *  A dispatcher will accept a task but will manage the insertion into the working-queue
 *  This way some fancy usage scenarios can be handled such as delayed execution or immediate
 *  posting.
 */

/*
 * std::function<void()> is a type that represents a callable object (function, lambda function,
 * functor) that takes no arguments and returns `void` i.e. does not return anything.
 *
 * This is often used to encapsulate and store functions or function-like objects with this
 * signature. The question is, is this useful?
 */
using Runnable = std::function<void()>;

// Explanation of general type T
// We have a problem of type T, as long as that problem is
// callable and each problem has the same structure then
// we can insert any problem into the queue. As a problem is a
// functor it is easy to satisfy the compiler.
template <typename M, typename ValueType>
class CLooper {
public:
    typedef typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType PrepReturnType;
    typedef SchedulerProblem<ValueType> Sch;
    typedef Eigen::SparseMatrix<ValueType, Eigen::RowMajor> SpMat;

    CLooper(uint id) : id(id), mRunning(false), mAbortRequested(false), mRunnables(),
                       mRunnablesMutex(), mDispatcher(std::shared_ptr<CDispatcher>(new CDispatcher(*this))),
                       mBusy(false), expectedSolutions(0) {

    };

    CLooper(uint id_, ThreadSpecialisation spec, PrepReturnType const& model, std::vector<int> const& rowGroupIndices);


    // Copy denied, move to be implemented

    ~CLooper() {
        // called in case the looper goes out of scope.
        abortAndJoin();
    }

    // To be called once the looper should start looping
    bool run();

    // To be called to stop a thread
    bool stop();

    // Check if the thread is running
    bool running() const;

    // Check if the thread is busy computing
    bool busy() const;

    // Check if poison pill inserted
    bool getAbortRequested() const;

    // Send data
    void sendDataGPU(SpMat& matrix, std::vector<int> const& rowGroupIndices);

    void sendDataCPU(SpMat& matrix);

    // Return solutions from the thread
    std::vector<std::pair<int, double>> getSolution();

    // Computes the next problem
    boost::optional<Sch> next();

    // Flag to check if all tasks have been computed by the thread
    bool solutionsReady();

    // Task Dispatcher
    class CDispatcher {
        friend class CLooper; // Allow the dispatcher access to the private members

    public:

        // Idea: make this more generic and send problems of different structures
        // for example a MDP scheduler generation function vs a DTMC problem
        bool post(Sch &&aRunnable) {
            return mAssignedLooper.post(std::move(aRunnable));
        }

    private:
        explicit CDispatcher(CLooper &aLooper) : mAssignedLooper(aLooper) {};

        CLooper &mAssignedLooper;
    };

    std::shared_ptr<CDispatcher> getDispatcher() { return mDispatcher; };

    bool poolAbortAndJoin();

private:

    void abortAndJoin();

    // Implements a thread function
    void runFunc();

    bool post(Sch &&aRunnable);


    std::thread mThread;
    std::atomic_bool mRunning;
    std::atomic_bool mBusy;
    std::atomic_bool mAbortRequested;
    std::queue<Sch> mRunnables; /* This is just a standard queue it can take anything
                                  * probably the best thing to do is insert a class
                                  * The class could evn be a functor which calls its own
                                  * model checking operation
                                  */
    std::recursive_mutex mRunnablesMutex;
    std::shared_ptr<CDispatcher> mDispatcher;
    std::vector<std::pair<int, double>> solutions;
    uint expectedSolutions;
    uint id;
    hybrid::utilities::CuMDPMatrix<ValueType> cuTransitionMatrix;
    // TODO specialise the thread for serving GPU or CPU operations
    hybrid::ThreadSpecialisation threadType;
    const uint64_t m, n, k;
    std::vector<ValueType> rhoFlat;
    std::unique_ptr<storm::storage::SparseMatrix<ValueType>> P;
    std::vector<uint64_t> pi, stateIndices;
};

template <typename T, typename ValueType>
class CLooperPool {
public:
    CLooperPool(std::vector<std::unique_ptr<CLooper<T, ValueType>>>&& loopers): mLoopers(std::move(loopers)){};

    ~CLooperPool() {
        stop();
    }

    bool run();

    bool running();

    void stop();

    // TODO The CLooperPool is currently not fit for purpose because we really only need two threads
    //  in the thread pool. Essentially we exploit multithreading with the CPU through eigen on the CPU
    //  thread dispatcher and with cuda natively on the GPU thread
    void solve(std::vector<hybrid::SchedulerProblem<ValueType>> tasks);

    void collectSolutions();

    // Every problem needs to return the same configuration or overload get solutions
    std::vector<std::pair<uint, double>>& getSolutions();

    std::vector<std::shared_ptr<typename CLooper<T, ValueType>::CDispatcher>> getDispatchers();

private:
    std::vector<std::unique_ptr<T>>&& mLoopers;
    std::vector<std::pair<uint, double>> solutions;
};
}
#endif //MOPMC_LOOPER_H
