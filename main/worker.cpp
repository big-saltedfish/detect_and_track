#include "worker.h"

int track_target::TIME = 0;
Worker::Worker(string openfile, int _N_FRAME_A_DETECT,int _TIME, int _THREAD_NUM, int _MAXTARGET) : pool(_THREAD_NUM){
        N_FRAME_A_DETECT = _N_FRAME_A_DETECT;
        TIME = _TIME;
        THREAD_NUM =  _THREAD_NUM;
        MAXTARGET = _MAXTARGET;
        track_target::TIME = TIME;
        cap.open("2.mp4");
        make_window("predictions", 512, 512, 0);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);  //¿í¶È
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640); //žß¶È
        cap >> frame[0];
        cap >> frame[1];
        for (int i = 0; i < MAXTARGET; i++) {
            tg[i].ok = false;
            tg[i].tracker = trackertemp;
        }
        procTrack.MenberFunc = &Worker::track_in_thread;
        procDetect.MenberFunc = &Worker::detect_in_thread;
        procShow.MenberFunc = &Worker::show_in_thread;
        procRead.MenberFunc = &Worker::read_in_thread;
        procTrackSon.MenberFunc = &Worker::track_son_thread;
}

void Worker::draw_detection(Mat im) {
        for (int i = 0; i < MAXTARGET; i++)
           if (tg[i].ok) {
              char *labelstr = tg[i].labelstr;
               int width = im.rows * .006;
               Rect2d b = tg[i].result;
               rectangle (im, tg[i].result, Scalar(0, 255, 255), 10, 8, 0);
           }
    }

int Worker::detect_in_thread() {
    yolo.detect(frame[buff_index], tg, MAXTARGET);
    draw_detection(frame[buff_index]);
    toc = what_time_is_it_now() - tic;
    printf("detect:%.2f\n", 1 / toc);
    return 0;
}

int Worker::track_son_thread(int id) {
    tg[id].result = tg[id].tracker.update(frame[buff_index]);
    return 0;
}

int Worker::track_in_thread() {
    int i;
    std::vector< std::future<int> > results;
    for (i = 0; i < MAXTARGET; i++)
        if (tg[i].ok) {
            results.emplace_back(pool.enqueue(procTrackSon.ThreadProc, this, i));
        }
    for(auto && result: results)    //Í¨¹ýfuture.get()»ñÈ¡·µ»ØÖµ
        result.get();
    results.clear();
    draw_detection(frame[buff_index]);
    toc = what_time_is_it_now() - tic;
    printf("track:%.2f\n", 1 / toc);
    return 0;
}

int Worker::show_in_thread() {
    imshow("predictions", frame[(buff_index+2)%3]);
    waitKey(1);
    toc = what_time_is_it_now() - tic;
    printf("show:%.2f\n", 1 / toc);
    return 0;
}

int Worker::read_in_thread(){
    cap >> frame[(buff_index + 1) % 3];
    toc = what_time_is_it_now() - tic;
    printf("show:%.2f\n", 1 / toc);
    return 0;
}

void Worker::update(){
    tic = what_time_is_it_now();
    buff_index = (buff_index + 1) % 3;
    void *ptr = NULL;
    if (index == 0) {
        results.emplace_back(pool.enqueue(procDetect.ThreadProc, this));
    } else {
        results.emplace_back(pool.enqueue(procTrack.ThreadProc, this));
    }
    results.emplace_back(pool.enqueue(procShow.ThreadProc, this));
    results.emplace_back(pool.enqueue(procRead.ThreadProc, this));
    for(auto && result: results)
        result.get();
    results.clear();

    for (int i = 0; i < MAXTARGET; i++)
        if (tg[i].ok) {
            tg[i].time--;
            if (tg[i].time == 0)
                tg[i].ok = false;
        }
    index++;
    if (index == N_FRAME_A_DETECT)
        index = 0;
    toc = what_time_is_it_now() - tic;
    printf("all:%.2f\n\n\n", 1 / toc);
    all_time += 1 / toc;
    sum_of_frame++;
}