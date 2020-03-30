#ifndef YOLO_DETECTER_H_
#define YOLO_DETECTER_H_

#include "detecter.h"
#include "image.h"

using namespace cv;
extern "C" cv::Mat image_to_mat(image im);

class YoloDetecter : public Detecter
{
	int argc;
	char **argv;
	char st[4][30];
public:
    YoloDetecter()  {
    	argc = 4;
    	strcpy(&st[0][0], "./darknet");
        strcpy(&st[1][0], "detect");
    	strcpy(&st[2][0], "cfg/yolov3-tiny.cfg");
    	strcpy(&st[3][0], "yolov3-tiny.weights");
    	argv = (char**)&st[0][0];
    }
    virtual void  detect(cv::Mat, track_target*, int);
private:
    float overlap_rate(Rect2d a, Rect2d b) {
    return (a & b).area() / (a.area() + b.area() - (a & b).area());
    }
    void update_detection(Mat frame, image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, track_target* tg, int n);
};



#endif

