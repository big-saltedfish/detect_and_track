
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



using namespace std;
using namespace cv;


const int N_FRAME_A_DETECT = 20;     // 每mod帧，进行一帧检测
const int TIME = 20;   // 若跟踪目标连续TIME帧未被检测，则认为跟踪目标已丢失，放弃跟踪
const int THREAD_NUM = 8;  // 线程池线程队列数量
const int MAXTARGET = 8;
VideoCapture cap;

Mat frame[3];

int M;

double toc, tic;
int buff_index;

track_target tg[MAXTARGET];

int argcc;
char** argvv;
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;








ThreadPool pool(1);




float overlap_rate(Rect2d a, Rect2d b) {
    return (a & b).area() / (a.area() + b.area() - (a & b).area());
}



void draw_detection(Mat im) {
    for (int i = 0; i < MAXTARGET; i++)
        if (tg[i].ok) {
            char *labelstr = tg[i].labelstr;
            int width = im.rows * .006;
            Rect2d b = tg[i].result;
            rectangle (im, tg[i].result, Scalar(0, 255, 255), 10, 8, 0);
        }
}

YoloDetecter yolo;

int detect_in_thread(void *ptr) {
    yolo.detect(frame[buff_index], tg, MAXTARGET);
    draw_detection(frame[buff_index]);
    
    toc = what_time_is_it_now() - tic;
    printf("detect:%.2f\n", 1 / toc);
    return 0;
}


int track_son_thread(int id) {
    tg[id].result = tg[id].tracker.update(frame[buff_index]);
    return 0;
}

int track_in_thread(void *ptr) {
    int i;
    std::vector< std::future<int> > results;
    for (i = 0; i < MAXTARGET; i++)
        if (tg[i].ok) {
            results.emplace_back(pool.enqueue(track_son_thread, i));
        }
    for(auto && result: results)    //通过future.get()获取返回值
        result.get();
    results.clear();
    draw_detection(frame[buff_index]);
    toc = what_time_is_it_now() - tic;
    printf("track:%.2f\n", 1 / toc);
    return 0;
}

int show_in_thread(void *ptr) {
    cout << "????";
    imshow("predictions", frame[(buff_index+2)%3]);
    waitKey(1);
    toc = what_time_is_it_now() - tic;
    printf("show:%.2f\n", 1 / toc);
    return 0;
}


int main(int argc, char **argv) {
    argc = 4;
    argv[0] = "./darknet";
    argv[1] = "detect";
    argv[2] = "cfg/yolov3-tiny.cfg";
    argv[3] = "yolov3-tiny.weights";
    argcc = argc;
    argvv = argv;
    cap.open("b.MOV");
    make_window("predictions", 512, 512, 0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);  //宽度
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640); //高度


    cap >> frame[0];
    cap >> frame[1];

    KCFTracker trackertemp(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    for (int i = 0; i < MAXTARGET; i++) {
        tg[i].ok = false;
        tg[i].tracker = trackertemp;
    }

    Mat buff;
    double all_time = 0;
    int sum_of_frame = 0;

    std::vector< std::future<int> > results;
    while (1) {
        tic = what_time_is_it_now();
        buff_index = (buff_index + 1) % 3;
        void *ptr = NULL;
        if (M == 0) {
            results.emplace_back(pool.enqueue(detect_in_thread, ptr));
        } else {
            results.emplace_back(pool.enqueue(track_in_thread, ptr));
        }
        results.emplace_back(pool.enqueue(show_in_thread, ptr));
        cap >> frame[(buff_index + 1) % 3];
        
        toc = what_time_is_it_now() - tic;
        printf("read:%.2f\n", 1 / toc);

        for(auto && result: results)
            result.get();
        results.clear();

        for (int i = 0; i < MAXTARGET; i++)
            if (tg[i].ok) {
                tg[i].time--;
                if (tg[i].time == 0)
                    tg[i].ok = false;
            }
        M++;
        if (M == N_FRAME_A_DETECT)
            M = 0;

        toc = what_time_is_it_now() - tic;
        printf("all:%.2f\n\n\n", 1 / toc);
        all_time += 1 / toc;
        sum_of_frame++;
        if (sum_of_frame == 400)
            break;
    }
    printf("avl time: %.3f", all_time / sum_of_frame);
}
