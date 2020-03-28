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

using namespace std;
using namespace cv;

const int MOD = 4;     // ÿmod֡������һ֡���
const int MAXN = 8;    // ������Ŀ������
const int TIME = 10;   // ������Ŀ������TIME֡δ����⣬����Ϊ����Ŀ���Ѷ�ʧ����������
const int THREAD_NUM = 8;  // �̳߳��̶߳�������

VideoCapture cap;

Mat frame[3];
image img[3];        // ���߳�ͼƬ�������

int M;
int argcc;
char **argvv;
double toc, tic;
int buff_index;

struct track_target {
    bool ok;
    KCFTracker tracker;
    Rect result;
    float red;
    float green;
    float blue;
    float rgb[3];
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





class ThreadPool {
public:
    ThreadPool(size_t);    //���캯����size_t n ��ʾ������

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)   //����ܵ�����
    -> std::future<typename std::result_of<F(Args...)>::type>;  //����β���޶���  std future������ȡ�첽����Ľ��

    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;   //׷���߳�
    // the task queue
    std::queue< std::function<void()> > tasks;    //������У����ڴ��û�д���������ṩ�������

    // synchronization  ͬ����
    std::mutex queue_mutex;   //������
    std::condition_variable condition;   //����������
    bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads): stop(false) {
    for(size_t i = 0; i<threads; ++i)
        workers.emplace_back(     //����Ϊ����һ�����񣬼�����һ���߳�
        [this] {
        for(;;) {
            std::function<void()> task;   //�߳��еĺ�������
            {
                //���������ã���ʱ�����������ڣ�������lock��ʱ��
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                this->condition.wait(lock,
                [this] { return this->stop || !this->tasks.empty(); }); //��stop==false&&tasks.empty(),���̱߳����� !this->stop&&this->tasks.empty()
                if(this->stop && this->tasks.empty())
                    return;
                task = std::move(this->tasks.front());
                this->tasks.pop();

            }

            task(); //���ú��������к���
        }
    }
    );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)  //&& �����޶�������������ֵ���ã�  �˴���ʾ��������һ������
-> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    //packaged_task�Ƕ������һ���������ǿ��Ը��䴫��һ������������乹�졣֮������Ͷ�ݸ��κ��߳�ȥ��ɣ�ͨ��
//packaged_task.get_future()������ȡ��future����ȡ������ɺ�Ĳ���ֵ
    auto task = std::make_shared<std::packaged_task<return_type()> >(  //ָ��F����������ָ��
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)  //���ݺ������й���
                );
    //futureΪ������get_future��ȡ������ɺ�Ĳ���ֵ
    std::future<return_type> res = task->get_future();   //��ȡfuture�������task��״̬��Ϊready����������ǰ������
    {
        std::unique_lock<std::mutex> lock(queue_mutex);  //���ֻ����ԣ��������߳�ͬʱ����һ������

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() {
            (*task)();
        });  //��taskͶ�ݸ��߳�ȥ��ɣ�vectorβ��ѹ��
    }
    condition.notify_one();  //ѡ��һ��wait״̬���߳̽��л��ѣ���ʹ����ö����ϵ������������(�������߳��޷����ʶ���)
    return res;
}//notify_one���ܱ�֤��������߳�������Ҫ����������˿��ܲ�������

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();  //֪ͨ����wait״̬���߳̾�������Ŀ���Ȩ�����������߳�ִ��
    for(std::thread &worker: workers)
        worker.join(); //��Ϊ�̶߳���ʼ�����ˣ�����һ����ִ���꣬join�ɵȴ��߳�ִ����
}


ThreadPool pool(8);

//extern image mat_to_image(Mat m);
extern void make_window(char *name, int w, int h, int fullscreen);

IplImage *image_to_ipl(image im) {
    int x, y, c;
    IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (c = 0; c < im.c; ++c) {
                float val = im.data[c * im.h * im.w + y * im.w + x];
                disp->imageData[y * step + x * im.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return disp;
}
Mat image_to_mat(image im) {
    image copy = copy_image(im);
    constrain_image(copy);
    if (im.c == 3)
        rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

float overlap_rate(Rect2d a, Rect2d b) {
    return (a & b).area() / (a.area() + b.area() - (a & b).area());
}

void update_detection(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes) {
    int i, j;
    for (i = 0; i < num; ++i) {

        char labelstr[4096] = {0};
        int classs = -1;
        for (j = 0; j < classes; ++j) {
            if(strcmp(names[j], "car") != 0) {
                continue;
            }
            if (dets[i].prob[j] > thresh) {
                if (classs < 0) {
                    strcat(labelstr, names[j]);
                    classs = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                // if (M == 0)
                //     printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);
            }
        }
        if (classs >= 0) {
            int offset = classs * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];


            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
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
                                tg[i].blue = blue;
                                tg[i].green = green;
                                tg[i].red = red;
                                strcpy(tg[i].labelstr, labelstr);
                                tg[i].time = TIME;

                                break;
                            }
                    }
                }
            }

            if (alphabet) {

                image label = get_label(alphabet, labelstr, (im.h * .03));
                //draw_label(im, top + width, left, label, rgb);
                free_image(label);
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

            float rgb[3];

            //width = prob*20+2;
            float red = tg[i].red;
            float green = tg[i].green;
            float blue = tg[i].blue;
            rgb[0] = tg[i].red;
            rgb[1] = tg[i].green;
            rgb[2] = tg[i].blue;
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

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {

                image label = get_label(alphabet, labelstr, (im.h * .03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
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
    pthread_t thread[20];
    int i, temp[20];
    std::vector< std::future<int> > results;
    for (i = 0; i < MAXN; i++)
        if (tg[i].ok) {
            temp[i] = i;
            results.emplace_back(pool.enqueue(track_son_thread, i));
        }
    for(auto && result: results)    //ͨ��future.get()��ȡ����ֵ
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
    //��⺯����������

    cap.open("003.mp4");// cap.open("1.mp4");
    make_window("predictions", 512, 512, 0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);  //���
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640); //�߶�
    //ͼ��������ʾ����


    img[0] = get_image_from_stream(&cap);
    img[1] = get_image_from_stream(&cap);
    frame[0] = image_to_mat(img[0]);
    frame[1] = image_to_mat(img[1]);
    //ͼƬ����س�ʼ��

    KCFTracker trackertemp(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    for (int i = 0; i < MAXN; i++) {
        tg[i].ok = false;
        tg[i].tracker = trackertemp;
    }
    //���ٳ�ʼ��

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
        frame[(buff_index + 1) % 3] = image_to_mat(img[(buff_index + 1) % 3]);
        toc = what_time_is_it_now() - tic;
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
