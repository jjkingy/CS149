#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}


/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    this->max_threads = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    //利用原子操作保证安全性
    std::atomic<int> task_id(0);

    auto func = [&]() {
        while(true) {
            int cur_task_id = task_id.fetch_add(1);
            if(cur_task_id >= num_total_tasks) {
                return;
            }
            runnable->runTask(cur_task_id, num_total_tasks);
        }
    }

    std::vector<std::thread> threads;
    for(int i = 0; i < max_threads; i++) {
        threads.push_back(std::thread(func));
    }

    for(auto& thread : threads) {
        thread.join();
    }


    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    this->max_threads = num_threads;
    for(int i = 0; i < this->max_threads; i++) {
        this->thread_pool.emplace_back(std::thread([this]() {worker()}));
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    is_terminate = true;
    for (auto& t : thread_pool) {
        if (t.joinable()) {
            t.join();
        }
    }
}


void TaskSystemParallelThreadPoolSpinning::worker() {
    while(!is_terminate) {
        std::unique_lock<std::mutex> lock(m_mutex); //构造时立即加锁
        if(left_task_num <= 0) {
            lock.unlock();
            std::this_thread::yield();
            continue;   //直接跳过
        }

        int task_id = total_task_num - left_task_num;
        left_task_num--;
        lock.unlock();

        runner->runTask(task_id, total_task_num);
        finished_task_num++;
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
    finished_task_num = 0;
    this->total_task_num = num_total_tasks;
    runner = runnable;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        left_task_num = num_total_tasks;
    }

    while(finished_task_num < num_total_tasks) {
        std::this_thread::yield();
    }

}



TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

/*
    调用过程
    主线程                         worker1            worker2           ...
        |                               |                   |
        |----- new() -----------------> |                   |
        |                               |--- sleep -------->| 
        |                               |                   |
        |--- run() --- set tasks ------>|                   |
        |--- notify_all() ------------> wake up ----------> wake up
        |                               |--- run task ---->|
        |                               |--- run task ---->|
        |                               |                   |
    wait for finish <-------------------- notify_one <-----|
        |                               |                   |
    run() return                      loop again or exit
        |                               |                   |
    析构 notify_all -----------------> wake up, exit
        |                               |                   |
    析构 join() <-------------------- 线程退出

*/





const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

// TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
//     //
//     // TODO: CS149 student implementations may decide to perform setup
//     // operations (such as thread pool construction) here.
//     // Implementations are free to add new class member variables
//     // (requiring changes to tasksys.h).
//     //
//     this->max_threads = num_threads;
//     for(int i = 0; i < this->max_threads; i++) {
//         this->thread_pool.emplace_back(std::thread([this]() {worker()}));
//     }
// }

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads), num_threads(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    stop = false;
    total_task_num = left_task_num = 0;

    for (int i = 0; i < this->num_threads; ++i) {
        workers.push_back(std::thread([this]() { worker(); }));
    }
}


void TaskSystemParallelThreadPoolSleeping::worker() {
    while(true) {
        std::unique_lock<std::mutex> lock_worker(mtx_worker);
        auto wait_func = [this] () {
            return this->stop || this->left_task_num > 0;
        };
        cv_work.wait(lock_worker, wait_func);

        if(stop && left_task_num == 0) {
            break;
        }

        int task_id = total_task_num - left_task_num;
        left_task_num--;
        lock_worker.unlock();

        runner->runTask(task_id, this->total_task_num);

        {
            std::lock_guard<std::mutex> lock(mtx_finish);
            finished_task_num++;
            if(finished_task_num == total_task_num) {
                cv_finished.notify_one();
            }
        }
    }

}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    {
        std::lock_guard<std::mutex> lock(mtx_worker);
        stop = true;  // 设置退出标志
    }
    cv_worker.notify_all();  // 唤醒所有等待的工作线程

    for (auto& t : thread_pool) {
        if (t.joinable()) {
            t.join();  // 等待所有线程安全退出
        }
    }
}


void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
    runner = runnable;
    total_task_num = num_total_tasks;
    finished_task_num = 0;

    {
        std::lock_guard<std::mutex> lock(mtx_worker);
        left_task_num = total_task_num;
    }

    cv_worker.notify_all();

    std::unique_lock<std::mutex> lock(mtx_finish);
    auto wait_func = [this] () {
        return this->finished_task_num == this->total_task_num;
    };
    cv_finish.wait(lock, wait_func);
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
