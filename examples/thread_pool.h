#include <vector>
#include <chrono>
#include <stdio.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <algorithm>


class ThreadPool {
public:
    ThreadPool(size_t);    //¹¹Ôìº¯Êý£¬size_t n ±íÊŸÁ¬œÓÊý

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)   //ÈÎÎñ¹ÜµÀº¯Êý
    -> std::future<typename std::result_of<F(Args...)>::type>;  //ÀûÓÃÎ²ÖÃÏÞ¶š·û  std futureÓÃÀŽ»ñÈ¡Òì²œÈÎÎñµÄœá¹û

    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;   //×·×ÙÏß³Ì
    // the task queue
    std::queue< std::function<void()> > tasks;    //ÈÎÎñ¶ÓÁÐ£¬ÓÃÓÚŽæ·ÅÃ»ÓÐŽŠÀíµÄÈÎÎñ¡£Ìá¹©»º³å»úÖÆ

    // synchronization  Í¬²œ£¿
    std::mutex queue_mutex;   //»¥³âËø
    std::condition_variable condition;   //ÌõŒþ±äÁ¿£¿
    bool stop;
};
