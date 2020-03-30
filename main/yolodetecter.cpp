#include "yolodetecter.h"
#include <string.h>
#include "darknet.h"

extern image im;
extern int nboxes;
extern char *name_list;
extern char **names;
extern image **alphabet;
extern detection *dets;
extern "C" image mat_to_image(cv::Mat);
extern const int TIME = 20;
extern int argcc;
extern char** argvv;
using namespace cv;
image img;

void  YoloDetecter::detect(cv::Mat frame, track_target* target, int n){
    img = mat_to_image(frame);
    yolomain(argc, argv, img);
    update_detection(frame, im, dets, nboxes, 0.5, names, alphabet, 80, target, n);
    free_detections(dets, nboxes);
    free_image(img);
}

void YoloDetecter::update_detection(Mat frame, image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, track_target* tg, int n) {
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
            if (roi.width > 0 && roi.height > 0) {
                bool flag = false;
                for (int i = 0; i < n; i++) {
                    if (tg[i].ok && overlap_rate(roi, tg[i].result) > 0.3) {
                        tg[i].tracker.init(roi, frame);
                        tg[i].time = TIME;
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    for (int i = 0; i < n; i++)
                        if (!tg[i].ok) {
                            tg[i].ok = true;
                            tg[i].tracker.init(roi, frame);
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