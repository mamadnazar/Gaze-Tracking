#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{
    // Face Detection
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));
    
    // Eye Detection
    if (faces.size() == 0) {
        //std::cout << "Face not detected" << std::endl;
        return; // none face was detected
    }
    cv::Mat face = frame(faces[0]); // crop the face
    std::vector<cv::Rect> eyes;
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20)); // same thing as above
    
    // Draw Regions
    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
    if (eyes.size() != 2) {
        //std::cout << "Eyes not detected" << std::endl;
        return; // both eyes were not detected
    }
    for (cv::Rect &eye : eyes)
    {
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
}

int main()
{
    // Trained Models for faces and eyes
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;
    if (!faceCascade.load("./haarcascade_frontalface_alt.xml"))
    {
        std::cerr << "Could not load face detector." << std::endl;
        return -1;
    }
    if (!eyeCascade.load("./haarcascade_eye_tree_eyeglasses.xml"))
    {
        std::cerr << "Could not load eye detector." << std::endl;
        return -1;
    }
    
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
        detectEyes(frame, faceCascade, eyeCascade);
        cv::imshow("Webcam", frame); // displays the Mat
        if (cv::waitKey(30) >= 0) break; // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
    }
    return 0;
}
