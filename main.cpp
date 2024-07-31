#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;

Mat imgOriginal, imgGray, imgCanny, imgThre, imgBlur, imgDil, imgWarp, imgErode, imgCrop;
vector<Point> initialPoints, docPoints;
float w = 420, h = 596;

Mat preProcessing(Mat img)
{
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(7, 7), 5, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);

    return imgDil;
}

vector<Point> getContours(Mat image) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());

    vector<Point> biggest;
    int maxArea = 0;

    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);

        if (area > 1000)
        {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            if (area > maxArea && conPoly[i].size() == 4)
            {
                drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
                biggest = { conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3] };
                maxArea = area;
            }
        }
    }
    return biggest;
}

void drawPoints(vector<Point> points, Scalar color)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(imgOriginal, points[i], 5, color, FILLED);
        putText(imgOriginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, color, 2);
    }
}

vector<Point> reorder(vector<Point> points)
{
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;
    for (int i = 0; i < 4; i++)
    {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }

    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 1
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 2
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 3

    return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
    Point2f src[4] = { points[0],points[1],points[2],points[3] };
    Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));

    return imgWarp;
}

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open the webcam" << endl;
        return -1;
    }

    while (true) {
        cap.read(imgOriginal);
        if (imgOriginal.empty()) {
            cout << "Error: Could not capture frame" << endl;
            break;
        }

        imgThre = preProcessing(imgOriginal);
        initialPoints = getContours(imgThre);

        if (initialPoints.size() == 4) {
            docPoints = reorder(initialPoints);
            imgWarp = getWarp(imgOriginal, docPoints, w, h);
            int cropVal = 5;
            Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
            imgCrop = imgWarp(roi);

            // Save the cropped document image to the Downloads folder
            std::string downloadsPath = getenv("HOME");
            std::string savePath = downloadsPath + "/Downloads/scanned_document.png";
            imwrite(savePath, imgCrop);

            imshow("Image Crop", imgCrop);
        }

        imshow("Image", imgOriginal);
        if (waitKey(1) == 27) { // Exit if 'ESC' is pressed
            break;
        }
    }
    return 0;
}
