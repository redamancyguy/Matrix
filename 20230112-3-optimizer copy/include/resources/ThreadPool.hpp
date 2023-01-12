#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <condition_variable>
#include <thread>
#include <functional>
#include <stdexcept>


//max capacity of the pool
#define  THREAD_POOL_MAX_NUM 32
//can the number of worker threads increases automatically
//#define  THREAD_POOL_AUTO_GROW

//线程池,可以提交变参函数或拉姆达表达式的匿名函数执行,可以获取执行返回值
//不直接支持类成员函数, 支持类静态成员函数或全局函数,Opteron()函数等
class ThreadPool {
public:
    enum ShutdownMode{
        immediately = 0,
        gracefully = 1,
    };
private:
    unsigned short _init_size;       //初始化线程数量
    using Task = std::function<void()>; //定义类型
    std::vector<std::thread> _pool;          //线程池
    std::queue<Task> _tasks;            //任务队列
    std::mutex _lock;                   //任务队列同步锁
#ifdef THREAD_POOL_AUTO_GROW
    std::mutex _lockGrow;               //线程池增长同步锁
#endif // !THREAD_POOL_AUTO_GROW
    std::condition_variable _task_cv;   //条件阻塞
    std::atomic<bool> _run{true};     //线程池是否执行
    std::atomic<bool> _destroyed{false};
    std::atomic<std::size_t> _idl_thread_number{0};  //空闲线程数量

public:
    inline explicit ThreadPool(unsigned short size = 4) : _init_size(size) { add_threads(size); }

    inline ~ThreadPool() {
        destroy();
    }

public:
    void destroy(ShutdownMode mode = gracefully) {
        if(_destroyed){
            return;
        }
        _run = false;
        switch(mode){
            case immediately:{
                for (std::thread &thread: _pool) {
                    thread.detach();
                }
                break;
            }
            case gracefully:{}
            default:{
                _run = false;
                _task_cv.notify_all();
                for (std::thread &thread: _pool) {
                    if (thread.joinable())
                        thread.join(); //the task must can be finished
                }
                break;
            }
        }
        _destroyed = true;
    }
    // 提交一个任务
    // 调用.get()获取返回值会等待任务执行完,获取返回值
    // 有两种方法可以实现调用类成员，
    // 一种是使用   bind： .enqueue(std::bind(&Dog::sayHello, &dog));
    // 一种是用   mem_fn： .enqueue(std::mem_fn(&Dog::sayHello), this)

    template<class Fp, class ...BoundArgs>
    inline auto enqueue(Fp &&f, BoundArgs &&... args)
    -> std::future<decltype(f(args...))> {
        if (!_run)
            throw std::runtime_error("ThreadPool is stopped.");

        using RetType = decltype(f(args...)); // typename std::result_of<_Fp(_BoundArgs...)>::type
        auto task = std::make_shared<std::packaged_task<RetType()>>(
                std::bind(std::forward<Fp>(f), std::forward<BoundArgs>(args)...)
        ); // bind the function entrance and args
        std::future<RetType> future = task->get_future();
        {    // add a task to queue
            std::lock_guard<std::mutex> lock(_lock);//Lock a code block
            _tasks.emplace([task]() { // push a task at the end of the queue
                (*task)();
            });
        }
#ifdef THREAD_POOL_AUTO_GROW
        if (_idl_thread_number < 1 && _pool.size() < THREAD_POOL_MAX_NUM)
            addThread(1);
#endif // !THREAD_POOL_AUTO_GROW
        _task_cv.notify_one();//week up a worker thread

        return future;
    }

    //count idle threads
    std::size_t idle_Count() { return _idl_thread_number; }

    //count threads
    std::size_t thread_Count() { return _pool.size(); }

#ifndef THREAD_POOL_AUTO_GROW
private:
#endif // !THREAD_POOL_AUTO_GROW

    //add quantity of threads
    void add_threads(unsigned short quantity) {
#ifdef THREAD_POOL_AUTO_GROW
        if (!_run)
            throw std::runtime_error("Grow on ThreadPool is stopped.");
        std::unique_lock<std::mutex> lockGrow{_lockGrow}; //Automatic growth
#endif // !THREAD_POOL_AUTO_GROW
        for (; _pool.size() < THREAD_POOL_MAX_NUM && quantity > 0; --quantity) {   //add threads no more than max_num
            _pool.emplace_back([this] { //worker thread(function)
                Task task; // an operator to run task
                while (true) //防止 _run==false 时立即结束,此时任务队列可能不为空
                {
                    {
                        // unique_lock can unlock or lock at any time
                        std::unique_lock<std::mutex> lock{_lock};
                        _task_cv.wait(lock, [this] {
                            // wait until task_queue is not empty and the pool not stop
                            return !_run || !_tasks.empty();
                        });
                        if (!_run && _tasks.empty())
                            return;
                        _idl_thread_number--;
                        task = std::move(_tasks.front()); // take out a task
                        _tasks.pop();
                    }
                    task();//execute the task
#ifdef THREAD_POOL_AUTO_GROW
                    if (_idl_thread_number > 0 && _pool.size() > _init_size) //automatic release
                        return;
#endif // !THREAD_POOL_AUTO_GROW
                    {
                        std::unique_lock<std::mutex> lock{_lock};
                        _idl_thread_number++;
                    }
                }
            });
            {
                std::unique_lock<std::mutex> lock{_lock};
                _idl_thread_number++;
            }
        }
    }
};

#endif  //THREAD_POOL_H
