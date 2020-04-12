#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <sys/time.h>
#include "kcftracker.hpp"
#include "image.h"
#include "thread_pool.hpp"
#include "framework.h"
#include "yolodetecter.h"
#include <string>

using namespace std;
using namespace cv;

class Worker{
private:
    int N_FRAME_A_DETECT;    //every N_FRAME_A_DETECT frame,a frame wile be detected
    int TIME;   //if TIME frame the traget is not be detected,tracker give up it
    int THREAD_NUM;  //the number of threads in hread pool
    int MAXTARGET;  //max number of targets 


public:
    Worker(string openfile, int, int, int, int);
    void update();

private:
    VideoCapture cap;
    Mat frame[3];
    int index;
    double toc, tic;
    int buff_index;
    KCFTracker trackertemp;
    track_target tg[100];
    ThreadPool pool;
    YoloDetecter yolo;
    Mat buff;
    double all_time = 0;
    int sum_of_frame = 0;
    union{
        int (Worker::*MenberFunc)();
        int (*ThreadProc)(void *);
    }procRead, procShow, procDetect, procTrack;
    union{
        int (Worker::*MenberFunc)(int);
        int (*ThreadProc)(void *, int);
    }procTrackSon;

    std::vector< std::future<int> > results;
        float overlap_rate(Rect2d a, Rect2d b) {
     return (a & b).area() / (a.area() + b.area() - (a & b).area());
    }
    void draw_detection(Mat im);
    int detect_in_thread();
    int track_son_thread(int id);
    int track_in_thread();
    int show_in_thread();
    int read_in_thread();
};