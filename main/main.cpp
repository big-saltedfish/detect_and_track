#include "darknet.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <sys/time.h>
#include "image.h"
#include "kcftracker.hpp"
#include "image.h"
#include "thread_pool.hpp"





using namespace std;
using namespace cv;

const int MOD = 4;     // 每mod帧，进行一帧检测
const int MAXN = 8;    // 最大跟踪目标数量
const int TIME = 10;   // 若跟踪目标连续TIME帧未被检测，则认为跟踪目标已丢失，放弃跟踪
const int THREAD_NUM = 8;  // 线程池线程队列数量

VideoCapture cap;

Mat frame[3];
image img[3];        // 多线程图片缓存队列

int M;
int argcc;
char **argvv;
double toc, tic;
int buff_index;

struct track_target {
    bool ok;
    KCFTracker tracker;
    Rect result;
    char labelstr[4096];
    int time;
} tg[20];

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

extern image im;
extern int nboxes;
//extern list *options;
extern char *name_list;
extern char **names;
extern image **alphabet;
extern detection *dets;



extern void make_window(char *name, int w, int h, int fullscreen);
extern "C" IplImage *image_to_ipl(image im);
extern "C" Mat image_to_mat(image im);



ThreadPool pool(8);




float overlap_rate(Rect2d a, Rect2d b) {
    return (a & b).area() / (a.area() + b.area() - (a & b).area());
}

void update_detection(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes) {
    int i, j;
    for (i = 0; i < num; ++i) {
        char labelstr[4096] = {0};
        int classs = -1;
        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                if (classs < 0) {
                    strcat(labelstr, names[j]);
                    classs = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
            }
        }
        if (classs >= 0) {
            box b = dets[i].bbox;
            int left = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top = (b.y - b.h / 2.) * im.h;
            int bot = (b.y + b.h / 2.) * im.h;

            if (left < 0)
                left = 0;
            if (right > im.w - 1)
                right = im.w - 1;
            if (top < 0)
                top = 0;
            if (bot > im.h - 1)
                bot = im.h - 1;
            Rect2d roi = Rect2d(left, top, right - left, bot - top);

            if (M == 0) {
                if (roi.width > 0 && roi.height > 0) {
                    bool flag = false;
                    for (int i = 0; i < MAXN; i++) {
                        if (tg[i].ok && overlap_rate(roi, tg[i].result) > 0.3) {
                            tg[i].tracker.init(roi, frame[buff_index]);
                            tg[i].time = TIME;
                            flag = true;
                            break;
                        }
                    }
                    if (!flag) {
                        for (int i = 0; i < MAXN; i++)
                            if (!tg[i].ok) {
                                tg[i].ok = true;
                                tg[i].tracker.init(roi, frame[buff_index]);
                                tg[i].result = roi;
                                strcpy(tg[i].labelstr, labelstr);
                                tg[i].time = TIME;
                                break;
                            }
                    }
                }
            }
        }
    }
}

void draw_detection(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes) {
    int i, j;
    int sum = 0;
    for (int i = 0; i < MAXN; i++)
        if (tg[i].ok) {
            char *labelstr = tg[i].labelstr;
            int width = im.h * .006;
            Rect2d b = tg[i].result;
            int left = b.x;
            int right = b.x + b.width;
            int top = b.y;
            int bot = b.y + b.height;
            if (left < 0)
                left = 0;
            if (right > im.w - 1)
                right = im.w - 1;
            if (top < 0)
                top = 0;
            if (bot > im.h - 1)
                bot = im.h - 1;
            draw_box_width(im, left, top, right, bot, width, 0.8, 1.0, 0);
        }
}


int detect_in_thread(void *ptr) {
    yolomain(argcc, argvv, img[buff_index]);
    update_detection(im, dets, nboxes, 0.5, names, alphabet, 80);
    draw_detection(img[buff_index], dets, nboxes, 0.5, names, alphabet, 80);
    free_detections(dets, nboxes);
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
    for (i = 0; i < MAXN; i++)
        if (tg[i].ok) {
            results.emplace_back(pool.enqueue(track_son_thread, i));
        }
    for(auto && result: results)    //通过future.get()获取返回值
        result.get();
    results.clear();
    draw_detection(img[buff_index], dets, nboxes, 0.5, names, alphabet, 80);
    toc = what_time_is_it_now() - tic;
    printf("track:%.2f\n", 1 / toc);
    return 0;
}

int show_in_thread(void *ptr) {
    show_image(img[(buff_index+2)%3], "predictions", 1);
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
    //检测函数参数设置

    cap.open("b.MOV");// cap.open("1.mp4");
    make_window("predictions", 512, 512, 0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);  //宽度
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640); //高度
    //图像输入显示设置


    img[0] = get_image_from_stream(&cap);
    img[1] = get_image_from_stream(&cap);
    frame[0] = image_to_mat(img[0]);
    frame[1] = image_to_mat(img[1]);
    //图片缓存池初始化

    KCFTracker trackertemp(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    for (int i = 0; i < MAXN; i++) {
        tg[i].ok = false;
        tg[i].tracker = trackertemp;
    }
    //跟踪初始化

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
        free_image(img[(buff_index + 1) % 3]);
        img[(buff_index + 1) % 3] = get_image_from_stream(&cap);
        toc = what_time_is_it_now() - tic;
        printf("read:%.2f\n", 1 / toc);
        frame[(buff_index + 1) % 3] = image_to_mat(img[(buff_index + 1) % 3]);
        toc = what_time_is_it_now() - toc;
        printf("read:%.2f\n", 1 / toc);

        for(auto && result: results)
            result.get();
        results.clear();

        for (int i = 0; i < MAXN; i++)
            if (tg[i].ok) {
                tg[i].time--;
                if (tg[i].time == 0)
                    tg[i].ok = false;
            }
        M++;
        if (M == MOD)
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
