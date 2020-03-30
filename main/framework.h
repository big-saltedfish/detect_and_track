#ifndef FRAME_WORK_H_
#define FRAME_WORK_H_


#include "kcftracker.hpp"
#include <opencv2/opencv.hpp>


struct track_target {
    bool ok;
    KCFTracker tracker;
    cv::Rect result;
    char labelstr[4096];
    int time;
};



#endif