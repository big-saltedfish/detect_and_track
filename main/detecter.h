#ifndef DETECTER_H_
#define DETECTER_H_

#include "framework.h"
#include <opencv2/opencv.hpp>


class Detecter
{
public:
    Detecter() {};
    virtual  ~Detecter() {};
    virtual void  detect(cv::Mat, track_target*, int)=0;

};

#endif