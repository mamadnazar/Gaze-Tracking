// g++ $(pkg-config --cflags --libs opencv4) -std=c++11 gazeTracking.cpp -o gazeTracking

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

std::vector<cv::Point> rightEyeCenters;
std::vector<cv::Point> leftEyeCenters;

// Getting Left Eye
cv::Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
{
    int leftmost = 99999999;
    int leftmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x < leftmost)
        {
            leftmost = eyes[i].tl().x;
            leftmostIndex = i;
        }
    }
    return eyes[leftmostIndex];
}

// Getting Right Eye
cv::Rect getRightmostEye(std::vector<cv::Rect> &eyes)
{
    int rightmost = -999999;
    int rightmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].br().x > rightmost)
        {
            rightmost = eyes[i].br().x;
            rightmostIndex = i;
        }
    }
    return eyes[rightmostIndex];
}

// Getting Eyeball
cv::Vec3f getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
    std::vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++)
    {
        uchar *ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++)
        {
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++)
            {
                cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
                int radius = (int)std::round(circles[i][2]);
                if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
                {
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 9999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++)
    {
        if (sums[i] < smallestSum)
        {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

// Stabilizing iris detection
cv::Point stabilize(std::vector<cv::Point> &points, int windowSize)
{
    float sumX = 0;
    float sumY = 0;
    int count = 0;
    for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
    {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;
    }
    if (count > 0)
    {
        sumX /= count;
        sumY /= count;
    }
    return cv::Point(sumX, sumY);
}

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{
    // Face Detection
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(150, 150));
    
    // Eye Detection
    if (faces.size() == 0) {
        //std::cout << "Face not detected" << std::endl;
        return; // none face was detected
    }
    cv::Mat face = frame(faces[0]); // crop the face
    std::vector<cv::Rect> eyes;
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(15, 15)); // same thing as above
    
    // Draw Regions
    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 255, 255), 2);
    if (eyes.size() != 2) {
        //std::cout << "Eyes not detected" << std::endl;
        return; // both eyes were not detected
    }
    for (cv::Rect &eye : eyes)
    {
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
    
    // Detect and draw Left Iris
    cv::Rect eyeRect = getLeftmostEye(eyes);
    cv::Mat eye = face(eyeRect);
    cv::Mat grayscaleEye;
    cv::cvtColor(eye, grayscaleEye, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayscaleEye, grayscaleEye);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(grayscaleEye, circles, cv::HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);

    if (circles.size() > 0)
    {
        cv::Vec3f eyeball = getEyeball(eye, circles);
        // stabilizing
        cv::Point center(eyeball[0], eyeball[1]);
        leftEyeCenters.push_back(center);
        center = stabilize(leftEyeCenters, 5); // using the last 5
        
        // draw iris
        cv::Point centerPoint(faces[0].tl() + eyeRect.tl() + center);
        
        cv::Point leftPoint(centerPoint.x-5, centerPoint.y);
        cv::Point rightPoint(centerPoint.x+5, centerPoint.y);
        cv::Point upperPoint(centerPoint.x, centerPoint.y-5);
        cv::Point downPoint(centerPoint.x, centerPoint.y+5);
        cv::line(frame, leftPoint, rightPoint, cv::Scalar(0, 0, 255), 1, 8, 0);
        cv::line(frame, upperPoint, downPoint, cv::Scalar(0, 0, 255), 1, 8, 0);
        
        //int radius = (int)std::round(eyeball[2]);
        //cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
        //cv::circle(eye, center, radius, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Eye", eye);

    // Detect and draw rigth iris
    cv::Rect rightEyeRect = getRightmostEye(eyes);
    cv::Mat rightEye = face(rightEyeRect);
    cv::Mat grayscaleRightEye;
    cv::cvtColor(rightEye, grayscaleRightEye, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayscaleRightEye, grayscaleRightEye);
    std::vector<cv::Vec3f> rightEyeCircles;
    cv::HoughCircles(grayscaleRightEye, rightEyeCircles, cv::HOUGH_GRADIENT, 1, rightEye.cols / 8, 250, 15, rightEye.rows / 8, rightEye.rows / 3);

    if (rightEyeCircles.size() > 0)
    {
        cv::Vec3f rightEyeball = getEyeball(rightEye, rightEyeCircles);
        // stabilizing
        cv::Point rightEyeballCenter(rightEyeball[0], rightEyeball[1]);
        rightEyeCenters.push_back(rightEyeballCenter);
        rightEyeballCenter = stabilize(rightEyeCenters, 5); // we are using the last 5
        // draw iris
        cv::Point rightIrisCenterPoint(faces[0].tl() + rightEyeRect.tl() + rightEyeballCenter);

        cv::Point leftPoint(rightIrisCenterPoint.x-5, rightIrisCenterPoint.y);
        cv::Point rightPoint(rightIrisCenterPoint.x+5, rightIrisCenterPoint.y);
        cv::Point upperPoint(rightIrisCenterPoint.x, rightIrisCenterPoint.y-5);
        cv::Point downPoint(rightIrisCenterPoint.x, rightIrisCenterPoint.y+5);
        cv::line(frame, leftPoint, rightPoint, cv::Scalar(0, 0, 255), 1, 8, 0);
        cv::line(frame, upperPoint, downPoint, cv::Scalar(0, 0, 255), 1, 8, 0);

        //int radius = (int)std::round(rightEyeball[2]);
        //cv::circle(frame, faces[0].tl() + rightEyeRect.tl() + rightEyeballCenter, radius, cv::Scalar(0, 0, 255), 2);
        //cv::circle(rightEye, rightEyeballCenter, radius, cv::Scalar(255, 255, 255), 2);
    }
    //cv::imshow("Eye", rightEye);
    
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
