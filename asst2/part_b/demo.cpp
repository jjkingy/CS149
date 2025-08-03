#include "task_system.h"

// 记录每个 bulk task launch 的上下文
struct TaskGroup {
    TaskID id;
    IRunnable* runnable;
    int total_tasks;
    std::atomic<int> finished_tasks;

    std::unordered_set<TaskID> deps;
    std::unordered_set<TaskID> dependents; // 谁依赖我

    bool ready = false;

    std::mutex mtx;
    std::condition_variable cv;

    TaskGroup(TaskID id_, IRunnable* r, int n)
        : id(id_), runnable(r), total_tasks(n), finished_tasks(0) {}
};

class TaskSystemParallelThreadPoolSleeping : public ITaskSystem {
public:
    TaskSystemParallelThreadPoolSleeping(int num_threads);
    ~TaskSystemParallelThreadPoolSleeping();

    const char* name() override { return "Parallel + Thread Pool + Sleep"; }
    void run(IRunnable* runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void sync() override;

private:
    int num_threads;
    std::vector<std::thread> workers;
    std::atomic<bool> stop;

    std::queue<std::pair<TaskGroup*, int>> task_queue; // (任务组, task id)
    std::mutex queue_mtx;
    std::condition_variable queue_cv;

    std::atomic<TaskID> next_task_id;
    std::unordered_map<TaskID, TaskGroup*> task_groups;

    void worker_loop();
    void try_schedule_group(TaskGroup* group);
};

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads)
    : num_threads(num_threads), stop(false), next_task_id(0) {
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    stop = true;
    queue_cv.notify_all();
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    // 清理内存
    for (auto& pair : task_groups) {
        delete pair.second;
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    TaskID tid = next_task_id.fetch_add(1);
    auto* group = new TaskGroup(tid, runnable, num_total_tasks);

    // 记录依赖
    for (TaskID dep_id : deps) {
        group->deps.insert(dep_id);
        task_groups[dep_id]->dependents.insert(tid);
    }

    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        task_groups[tid] = group;
    }

    // 如果没有依赖，立即调度
    if (deps.empty()) {
        try_schedule_group(group);
    }

    return tid;
}

void TaskSystemParallelThreadPoolSleeping::try_schedule_group(TaskGroup* group) {
    {
        std::lock_guard<std::mutex> lock(group->mtx);
        group->ready = true;
    }
    // 将 group 的所有 tasks 放入全局队列
    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        for (int i = 0; i < group->total_tasks; ++i) {
            task_queue.emplace(group, i);
        }
    }
    queue_cv.notify_all();
}

void TaskSystemParallelThreadPoolSleeping::worker_loop() {
    while (true) {
        TaskGroup* group = nullptr;
        int task_id = 0;

        {
            std::unique_lock<std::mutex> lock(queue_mtx);
            queue_cv.wait(lock, [this] { return stop || !task_queue.empty(); });

            if (stop && task_queue.empty()) {
                return;
            }

            auto [g, tid] = task_queue.front();
            task_queue.pop();
            group = g;
            task_id = tid;
        }

        group->runnable->runTask(task_id, group->total_tasks);

        int finished = group->finished_tasks.fetch_add(1) + 1;

        // 如果此 group 所有任务完成，通知它的依赖者
        if (finished == group->total_tasks) {
            {
                std::lock_guard<std::mutex> lock(group->mtx);
            }
            group->cv.notify_all();

            // 遍历依赖这个 group 的 groups，看它们是否可以调度
            for (TaskID dep_id : group->dependents) {
                TaskGroup* dep_group = nullptr;
                {
                    std::lock_guard<std::mutex> lock(queue_mtx);
                    dep_group = task_groups[dep_id];
                    dep_group->deps.erase(group->id);
                    if (dep_group->deps.empty() && !dep_group->ready) {
                        try_schedule_group(dep_group);
                    }
                }
            }
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    for (auto& pair : task_groups) {
        TaskGroup* group = pair.second;
        std::unique_lock<std::mutex> lock(group->mtx);
        group->cv.wait(lock, [group]() { return group->finished_tasks == group->total_tasks; });
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    std::vector<TaskID> no_deps;
    TaskID tid = runAsyncWithDeps(runnable, num_total_tasks, no_deps);

    {
        std::unique_lock<std::mutex> lock(task_groups[tid]->mtx);
        task_groups[tid]->cv.wait(lock, [this, tid]() {
            return task_groups[tid]->finished_tasks == task_groups[tid]->total_tasks;
        });
    }
}
