#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

int main()
{
    // Reading WebCam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Webcam not detected." << std::endl;
        return -1;
    }
    cv::Mat frame;
    while (1)
    {
        cap >> frame; // outputs the webcam image to a Mat
        //if (!frame.data) break;
        cv::imshow("Webcam", frame); // displays the Mat
        if (cv::waitKey(30) >= 0) break; // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
    }
    return 0;
}

