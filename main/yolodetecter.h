#ifndef YOLO_DETECTER_H_
#define YOLO_DETECTER_H_

#include "detecter.h"
#include "image.h"

using namespace cv;
extern "C" cv::Mat image_to_mat(image im);

class YoloDetecter : public Detecter
{
	int argc;
	char st[4][30];
    char* argv[4];
public:
    YoloDetecter()  {
    	argc = 4;
        strcpy(st[0], "./darknet");
        strcpy(st[1], "detect");
        strcpy(st[2], "cfg/yolov3-tiny.cfg");
        strcpy(st[3], "yolov3-tiny.weights");
        argv[0] = st[0];
        argv[1] = st[1];
        argv[2] = st[2];
        argv[3] = st[3];
    }
    virtual void  detect(cv::Mat, track_target*, int);
private:
    float overlap_rate(Rect2d a, Rect2d b) {
    return (a & b).area() / (a.area() + b.area() - (a & b).area());
    }
    void update_detection(Mat frame, image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, track_target* tg, int n);
};



#endif

