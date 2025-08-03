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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    // 初始化线程池
    this->num_threads = num_threads;
    stop = false;
    next_task_id = 0;

    // 创建 num_threads 个工作线程并启动
    for(int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() { worker(); });
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    // 设置标志位，通知所有线程退出
    stop = true;
    queue_cv.notify_all(); // 唤醒所有等待的线程，防止它们一直阻塞

    // 等待线程结束
    for(auto& t : workers) {
        if(t.joinable()) 
            t.join();
    }

    // 清理所有 TaskGroup 内存
    for(auto& pair : task_groups) {
        delete pair.second;
    }
}

void TaskSystemParallelThreadPoolSleeping::worker() {
    while(true) {
        TaskGroup* group = nullptr;
        int task_id = 0;

        // 从任务队列中取任务
        // 任务队列中存的都是就绪任务,所以只要有就直接拿来用
        {
            std::unique_lock<std::mutex> lock(queue_mtx);
            // 等待直到有任务或 stop 为 true
            queue_cv.wait(lock, [this]() { return stop || !task_queue.empty(); });
            
            // 如果 stop 且任务队列为空，退出线程
            if(stop && task_queue.empty()) {
                return;
            }

            // 取出一个任务
            auto [g, tid] = task_queue.front();
            task_queue.pop();
            group = g;
            task_id = tid;
        }

        // 执行任务
        group->runnable->runTask(task_id, group->total_tasks);

        // 增加已完成任务计数
        int finished = group->finished_tasks.fetch_add(1) + 1;

        // 如果当前 group 的所有任务都完成
        if(finished == group->total_tasks) {
            {
                std::lock_guard<std::mutex> lock(group->mtx);
                // 这里锁只用来和 condition_variable 配合
            }
            group->cv.notify_all(); // 唤醒所有等待该 group 完成的线程

            // 遍历依赖此 group 的后续 groups
            for(TaskID dep_id : group->dependents) {
                TaskGroup* dep_group = nullptr;
                {
                    std::lock_guard<std::mutex> lock(queue_mtx);
                    dep_group = task_groups[dep_id];
                    dep_group->deps.erase(group->id); // 移除依赖

                    // 如果依赖都满足且尚未调度，调度它,调度它其实是把它加入任务队列
                    if(dep_group->deps.empty() && !dep_group->ready) {
                        enqueue_group(dep_group);
                    }
                }
            }
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::enqueue_group(TaskGroup* group) {
    {
        // 标记该 group 可调度
        std::lock_guard<std::mutex> lock(group->mtx);
        group->ready = true;
    }

    // 把该 group 所有 task 加入任务队列
    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        for(int i = 0; i < group->total_tasks; i++) {
            task_queue.emplace(group, i);
        }
    }
    queue_cv.notify_all(); // 唤醒线程来执行新任务
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    // 不带依赖的同步执行：提交后立即等待它完成
    std::vector<TaskID> no_deps;
    TaskID tid = runAsyncWithDeps(runnable, num_total_tasks, no_deps);

    // 等待该任务组完成
    {
        std::unique_lock<std::mutex> lock(task_groups[tid]->mtx);
        task_groups[tid]->cv.wait(lock, [this, tid]() {
            return task_groups[tid]->finished_tasks == task_groups[tid]->total_tasks;
        });
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {
    // 分配一个新的任务 ID
    TaskID tid = next_task_id.fetch_add(1);

    // 新建一个 TaskGroup 并设置基本信息
    auto* group = new TaskGroup(tid, runnable, num_total_tasks);

    // 记录依赖关系
    for(TaskID dep_id : deps) {
        group->deps.insert(dep_id);
        task_groups[dep_id]->dependents.insert(tid);
    }

    // 注册到全局任务组 map
    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        task_groups[tid] = group;
    }

    // 如果没有依赖，立即调度
    if(deps.empty()) {
        enqueue_group(group);
    }

    return tid;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    // 等待所有任务组都完成
    for(auto& pair : task_groups) {
        TaskGroup* group = pair.second;
        if (group->finished_tasks.load() < group->total_tasks) {
            std::unique_lock<std::mutex> lock(group->mtx);
            group->cv.wait(lock, [group]() {
                return group->finished_tasks == group->total_tasks;
            });
        }
    }
}



// TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
//     //
//     // TODO: CS149 student implementations may decide to perform setup
//     // operations (such as thread pool construction) here.
//     // Implementations are free to add new class member variables
//     // (requiring changes to tasksys.h).
//     //
//     this->num_threads = num_threads;
//     stop = false;
//     next_task_id = 0;
//     for(int i = 0; i < num_threads; ++i) {
//         workers.emplace_back([this] () { worker();});
//     }
// }

// TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
//     //
//     // TODO: CS149 student implementations may decide to perform cleanup
//     // operations (such as thread pool shutdown construction) here.
//     // Implementations are free to add new class member variables
//     // (requiring changes to tasksys.h).
//     //
//     stop = true;
//     queue_cv.notify_all();
//     for(auto& t : workers) {
//         if(t.joinable()) 
//         t.join();
//     }
//     //清理内存
//     for(auto& pair : task_groups) {
//         delete pair.second;
//     }
// }

// void TaskSystemParallelThreadPoolSleeping::worker() {
//     while(true) {
//         TaskGroup* group = nullptr;
//         int task_id = 0;

//         //从工作队列中取任务执行
//         {
//             std::unique_lock<std::mutex> lock(queue_mtx);
//             queue_cv.wait(lock, [this]() { return stop || !task_queue.empty();});
//             if(stop && task_queue.empty()) {
//                 return;
//             }

//             auto [g, tid] = task_queue.front();
//             task_queue.pop();
//             group = g;
//             task_id = tid;
//         }

//         group->runnable->runTask(task_id, group->total_tasks);

//         int finished = group->finished_tasks.fetch_add(1) + 1;

//         //此时如果group全部完成，则通知它的依赖者
//         if(finished == group->total_tasks) {
//             {
//                 std::lock_guard<std::mutex> lock(group->mtx);
//                  // nothing to do here, just for scope
//             }
//             group->cv.notify_all();

//             //遍历依赖这个group的groups 看他们是不是可以调度
//             for(TaskID dep_id : group->dependents) {
//                 TaskGroup* dep_group = nullptr;
//                 {
//                     std::lock_guard<std::mutex> lock(queue_mtx);
//                     dep_group = task_groups[dep_id];
//                     dep_group->deps.erase(group->id);
//                     if(dep_group->deps.empty() && !dep_group->ready) {
//                         enqueue_group(dep_group);
//                     }
//                 }
//             }
//         }

//     }
// }

// void TaskSystemParallelThreadPoolSleeping::enqueue_group(TaskGroup* group) {
//     {
//         std::lock_guard<std::mutex> lock(group->mtx);
//         group->ready = true;
//     }

//     //把group所有tasks放入任务队列
//     {
//         std::lock_guard<std::mutex> lock(queue_mtx);
//         for(int i = 0; i < group->total_tasks; i++) {
//             task_queue.emplace(group, i);
//         }
//     }
//     queue_cv.notify_all();
// }

// void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


//     //
//     // TODO: CS149 students will modify the implementation of this
//     // method in Parts A and B.  The implementation provided below runs all
//     // tasks sequentially on the calling thread.
//     //

//     std::vector<TaskID> no_deps;
//     TaskID tid = runAsyncWithDeps(runnable, num_total_tasks, no_deps);

//     {
//         std::unique_lock<std::mutex> lock(task_groups[tid]->mtx);
//         task_groups[tid]->cv.wait(lock, [this, tid] () {
//             return task_groups[tid]->finished_tasks == task_groups[tid]->total_tasks;

//         });
//     }

// }

// TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
//                                                     const std::vector<TaskID>& deps) {


//     //
//     // TODO: CS149 students will implement this method in Part B.
//     //                                                 

//     TaskID tid = next_task_id.fetch_add(1);
//     auto* group = new TaskGroup(tid, runnable, num_total_tasks);

//     //记录依赖
//     for(TaskID dep_id : deps) {
//         group->deps.insert(dep_id);
//         task_groups[dep_id]->dependents.insert(tid);
//     }

//     {
//         std::lock_guard<std::mutex> lock(queue_mtx);
//         task_groups[tid] = group;
//     }

//     if(deps.empty()) {
//         enqueue_group(group);
//     }

//     return tid;
// }

// void TaskSystemParallelThreadPoolSleeping::sync() {

//     //
//     // TODO: CS149 students will modify the implementation of this method in Part B.
//     //

//     for(auto& pair : task_groups) {
//         TaskGroup* group = pair.second;
//         if (group->finished_tasks.load() < group->total_tasks) {
//             std::unique_lock<std::mutex> lock(group->mtx);
//             group->cv.wait(lock, [group] () {
//                 return group->finished_tasks == group->total_tasks;
//             });
//         }
//     }

//     return;
// }


