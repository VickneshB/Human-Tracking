#include <iostream>
#include <opencv2/opencv.hpp>

class Detector
{
    enum Mode { Default, Daimler } m;
    cv::HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    std::string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    std::vector<cv::Rect> detect(cv::InputArray img)
    {
        std::vector<cv::Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(), 1.05, 2, true);
        return found;
    }
    void adjustRect(cv::Rect & r) const
    {
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    cv::Mat bgr_image(500, 500, CV_8UC3, cv::Scalar(0,0,0));

    cv::Size s = bgr_image.size();
    int height = s.height;
    int width = s.width;

    cv::Point2d pt1(0,0), pt2(500,500);
    cv::VideoCapture cap;

    cap.open(0);
    if(!cap.isOpened()){
        std::cout<<"Error";
        return -1;
    }

    int i = 1;
    Detector detector;

    do{
        cap.read(bgr_image);
        if (bgr_image.empty())
        {
            std::cout << "Finished reading: empty frame" << std::endl;
            break;
        }
        int64 t = cv::getTickCount();
        std::vector<cv::Rect> found = detector.detect(bgr_image);
        t = cv::getTickCount() - t;

        // show the window
        {
            std::ostringstream buf;
            buf << "Mode: " << detector.modeName() << " ||| " << "FPS: " << std::fixed << std::setprecision(1) << (cv::getTickFrequency() / (double)t);
            putText(bgr_image, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        for (std::vector<cv::Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            cv::Rect &r = *i;
            detector.adjustRect(r);
            rectangle(bgr_image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
        }


        cv::imshow("Image", bgr_image);
    }while(cv::waitKey(1) != 27);


    return 0;
}
