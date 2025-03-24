// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void addGray(int factor) {
	Mat img = imread("Images/cameraman.bmp",
		IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) + factor < 0) img.at<uchar>(i, j) = 0;
			else img.at<uchar>(i, j) = (img.at<uchar>(i, j) + factor) % 255;
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

void multiplyGray(int factor) {
	Mat img = imread("Images/cameraman.bmp",
		IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) * factor < 0) img.at<uchar>(i, j) = 0;
			else img.at<uchar>(i, j) = (img.at<uchar>(i, j) * factor) % 255;
		}
	}
	imshow("negative image", img);
	imwrite("images/name.bmp", img);
	waitKey(0);
}

void displayRGB() {
	Mat img = imread("Images/cameraman.bmp", IMREAD_COLOR);
	Mat blue(img.rows, img.cols, CV_8UC3);
	Mat red(img.rows, img.cols, CV_8UC3);
	Mat green(img.rows, img.cols, CV_8UC3);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			Vec3b pixel1 = img.at<Vec3b>(i, j);
			Vec3b pixel2 = img.at<Vec3b>(i, j);
			pixel[0] = pixel[2] = 0;
			green.at<Vec3b>(i, j) = pixel;
			pixel1[1] = pixel1[2] = 0;
			blue.at<Vec3b>(i, j) = pixel1;
			pixel2[0] = pixel2[1] = 0;
			red.at<Vec3b>(i, j) = pixel2;
		}
	}
	imshow("blue", blue);
	imshow("red", red);
	imshow("green", green);
	waitKey(0);
}

void convertGray() {
	Mat img = imread("Images/kids.bmp", IMREAD_COLOR);
	Mat bw(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			bw.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}
	}
	imshow("col", img);
	imshow("bw", bw);

	waitKey(0);
}

void binaryImg(int th) {
	Mat img = imread("Images/cameraman.bmp",
		IMREAD_GRAYSCALE);
	imshow("normal", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < th) img.at<uchar>(i, j) = 0;
			else img.at<uchar>(i, j) = 255;
		}
	}
	imshow("binaryimage", img);
	//imwrite("images/name.bmp", img);
	waitKey(0);
}

void RgbToHsv() {
	Mat img = imread("Images/flowers_24bits.bmp", IMREAD_COLOR);
	//Mat img = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);
	Mat H1(img.rows, img.cols, CV_8UC1);
	Mat S1(img.rows, img.cols, CV_8UC1);
	Mat V1(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			unsigned char B = pixel[0];
			unsigned char G = pixel[1];
			unsigned char R = pixel[2];
			float r = R / 255.0f;
			float g = G / 255.0f;
			float b = B / 255.0f;
			float M = max(r, max(g, b));
			float m = min(r, min(g, b));
			float C = M - m;
			float V = M;
			float H;
			float S;
			if (V != 0) {
				S = C / V;
			}
			else S = 0;
			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else H = 0;
			if (H < 0) H += 360;
			float H_norm = H * 255 / 360;
			float S_norm = S * 255;
			float V_norm = V * 255;

			H1.at<uchar>(i, j) = H_norm;
			S1.at<uchar>(i, j) = S_norm;
			V1.at<uchar>(i, j) = V_norm;
		}
	}
	imshow("H", H1);
	imshow("S", S1);
	imshow("V", V1);
	waitKey(0);
}

bool isInside(Mat img, int i, int j) {
	return i < img.rows && i >= 0 && j < img.cols && j >= 0;
}

void showSquare() {
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			Vec3b pixel = img.at< Vec3b>(i, j);
			unsigned char B = pixel[0];
			unsigned char G = pixel[1];
			unsigned char R = pixel[2];
			if (i < 128 && j < 128) {
				pixel[0] = pixel[1] = pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
			else if (i < 128 && j >= 128) {
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
			else if (i >= 128 && j < 128) {
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 0;
				img.at<Vec3b>(i, j) = pixel;
			}
			else if (i >= 128 && j >= 128) {
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
		}
	}
	imshow("square image", img);
	waitKey(0);
}

void inverseM() {
	float vals[9] = { 1, 2, 3,7,1,9,5,7,9 };
	Mat M(3, 3, CV_32FC1, vals); //4 parameter constructor
	Mat inv;
	std::cout << M << std::endl;
	waitKey(1000);
	invert(M, inv);
	std::cout << inv << std::endl;
	waitKey(10000);
}

void displayHistogram() {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int g[256] = { 0 };
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			g[img.at<uchar>(i, j)]++;
		}
	}
	showHistogram("myHisto", g, 256, 256);
	waitKey(0);
}

int* computeHistogram() {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int* g = (int*)malloc(256 * sizeof(int));
	for (int i = 0; i < 256; i++) g[i] = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			g[img.at<uchar>(i, j)]++;
		}
	}
	return g;
}

float* computePDF() {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int* g = (int*)malloc(256 * sizeof(int));
	for (int i = 0; i < 256; i++) g[i] = 0;
	float* pdf = (float*)malloc(256 * sizeof(float));
	for (int i = 0; i < 256; i++) pdf[i] = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			g[img.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			pdf[img.at<uchar>(i, j)] = (float)(g[img.at<uchar>(i, j)] / (float)(img.rows * img.cols));
		}
	}
	return pdf;
}

void computeHistogramWidth(int k) {
	int bin_width = 265.0 / k + 0.5;
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	int* g = (int*)malloc(k * sizeof(int));
	for (int i = 0; i < k; i++) g[i] = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			g[min(img.at<uchar>(i, j) / bin_width, k - 1)]++;
		}
	}
	showHistogram("ok", g, k, 256);
	waitKey(0);
}

void multiLevelTh(int WH, float TH, std::vector<int>& result) {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	float* pdf = computePDF();
	int window = 2 * WH + 1;
	//std::vector<int> result;
	result.push_back(0);
	for (int k = 0 + WH; k < 255 - WH; k++) {
		float avg = 0;
		float mx = 0;
		for (int j = k - WH; j < k + WH; j++) {
			avg += pdf[j];
			if (pdf[j] > mx) mx = pdf[j];
		}
		avg = avg / (2 * WH + 1);
		if (pdf[k] > avg + TH && pdf[k] >= mx) result.push_back(k);
	}
	result.push_back(255);
	int g[256] = { 0 };
	for (int i = 0; i < result.size(); i++)
		if (result.at(i)) g[result.at(i)] = pdf[result.at(i)] * img.rows * img.cols;
	showHistogram("his", g, 256, 256);
	waitKey(0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = img.at<uchar>(i, j);
			int mi = INT_MAX;
			int val = 0;
			for (int i = 0; i < result.size(); i++)
				if (abs(x - result.at(i)) < mi) mi = abs(x - result.at(i)), val = result.at(i);
			img.at<uchar>(i, j) = val;
		}
	}
	imshow("img", img);
	waitKey(0);
}

void multiLevelThHSV(int WH, float TH, std::vector<int>& result) {
	Mat img = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);
	imshow("original", img);
	Mat hsv_img, H;
	cvtColor(img, hsv_img, COLOR_BGR2HSV);
	extractChannel(hsv_img, H, 0);

	float pdf[256] = { 0 };
	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			pdf[H.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < 256; i++) pdf[i] /= (H.rows * H.cols);

	int window = 2 * WH + 1;
	result.push_back(0);
	for (int k = 0 + WH; k < 255 - WH; k++) {
		float avg = 0;
		float mx = 0;
		for (int j = k - WH; j < k + WH; j++) {
			avg += pdf[j];
			if (pdf[j] > mx) mx = pdf[j];
		}
		avg = avg / (2 * WH + 1);
		if (pdf[k] > avg + TH && pdf[k] >= mx) result.push_back(k);
	}
	result.push_back(255);

	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			int x = H.at<uchar>(i, j);
			int mi = INT_MAX;
			int val = 0;
			for (int i = 0; i < result.size(); i++)
				if (abs(x - result.at(i)) < mi) mi = abs(x - result.at(i)), val = result.at(i);
			H.at<uchar>(i, j) = val;
		}
	}
	extractChannel(hsv_img, H, 0);
	Mat img1;
	cvtColor(hsv_img, img1, COLOR_HSV2BGR);

	imshow("Modified", img1);
	waitKey(0);
}

void dither(int WH, float TH) {
	Mat img = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
	float* pdf = computePDF();
	int window = 2 * WH + 1;
	std::vector<int> result;
	result.push_back(0);
	for (int k = 0 + WH; k < 255 - WH; k++) {
		float avg = 0;
		float mx = 0;
		for (int j = k - WH; j < k + WH; j++) {
			avg += pdf[j];
			if (pdf[j] > mx) mx = pdf[j];
		}
		avg = avg / (2 * WH + 1);
		if (pdf[k] > avg + TH && pdf[k] >= mx) result.push_back(k);
	}
	result.push_back(255);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = img.at<uchar>(i, j);
			int mi = INT_MAX;
			int val = 0;
			for (int i = 0; i < result.size(); i++)
				if (abs(x - result.at(i)) < mi) mi = abs(x - result.at(i)), val = result.at(i);
			img.at<uchar>(i, j) = val;
			int error = x - val;
			if (isInside(img, i, j + 1))
				img.at<uchar>(i, j + 1) = max(0, min(img.at<uchar>(i, j + 1) + (7 * error / 16), 255));
			if (isInside(img, i + 1, j - 1))
				img.at<uchar>(i + 1, j - 1) = max(0, min(img.at<uchar>(i + 1, j - 1) + (3 * error / 16), 255));
			if (isInside(img, i + 1, j))
				img.at<uchar>(i + 1, j) = max(0, min(img.at<uchar>(i + 1, j) + (5 * error / 16), 255));
			if (isInside(img, i + 1, j + 1))
				img.at<uchar>(i + 1, j + 1) = max(0, min(img.at<uchar>(i + 1, j + 1) + (1 * error / 16), 255));
		}
	}
	imshow("img", img);
	waitKey(0);
}

//use cvtColor(src,hsvimg,COLOR_BGR2HSV from openCV for the last one, hue only up to 180!!!
//then use extractChannel(hsvImg,hch,0)

void CallBackFuncLab4(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	int rs = 0, cs = 0;
	double top_val = 0, bottom_left = 0, bottom_right = 0;
	double two_th = 0;
	double angle = 0.0;
	int c_min = INT_MAX, c_max = INT_MIN, r_min = INT_MAX, r_max = INT_MIN;
	if (event == EVENT_LBUTTONDOWN)
	{
		int area = 0;
		Vec3b col = (*src).at<Vec3b>(y, x);
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++)
			{
				if ((*src).at<Vec3b>(i, j) == col)
				{
					area++;
					rs += i;
					cs += j;
					if (j > c_max) c_max = j;
					if (j < c_min) c_min = j;
					if (i > r_max) r_max = i;
					if (i < r_min) r_min = i;
				}
			}
		}
		printf("Area: %d\n", area);
		std::cout << "Center of mass: r: " << rs / area << " c: " << cs / area;
		rs = rs / area;
		cs = cs / area;
		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++)
			{
				if ((*src).at<Vec3b>(i, j) == col)
				{
					top_val += ((i - rs) * (j - cs));
					bottom_left += ((j - cs) * (j - cs));
					bottom_right += ((i - rs) * (i - rs));
				}
			}
		}
		double top = 2 * top_val;
		double bottom = bottom_left - bottom_right;
		angle = atan2(2 * top_val, bottom_left - bottom_right) / 2 * 180.0 / PI;
		std::cout << std::endl << "Axis of elongation: " << angle << std::endl;

		int per = 0;
		Mat test(src->rows, src->cols, CV_8UC3);
		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++)
			{
				if ((*src).at<Vec3b>(i, j) == col) {
					if (isInside(*src, i, j - 1) && (*src).at<Vec3b>(i, j - 1) != col) per++, test.at<Vec3b>(i, j - 1) = col;
					else if (isInside(*src, i - 1, j - 1) && (*src).at<Vec3b>(i - 1, j - 1) != col) per++, test.at<Vec3b>(i - 1, j - 1) = col;
					else if (isInside(*src, i + 1, j - 1) && (*src).at<Vec3b>(i + 1, j - 1) != col) per++, test.at<Vec3b>(i + 1, j - 1) = col;
					else if (isInside(*src, i + 1, j) && (*src).at<Vec3b>(i + 1, j) != col) per++, test.at<Vec3b>(i + 1, j) = col;
					else if (isInside(*src, i - 1, j) && (*src).at<Vec3b>(i - 1, j) != col) per++, test.at<Vec3b>(i - 1, j) = col;
					else if (isInside(*src, i - 1, j + 1) && (*src).at<Vec3b>(i - 1, j + 1) != col) per++, test.at<Vec3b>(i - 1, j + 1) = col;
					else if (isInside(*src, i, j + 1) && (*src).at<Vec3b>(i, j + 1) != col) per++, test.at<Vec3b>(i, j + 1) = col;
					else if (isInside(*src, i + 1, j + 1) && (*src).at<Vec3b>(i + 1, j + 1) != col) per++, test.at<Vec3b>(i + 1, j + 1) = col;
				}
			}
		}
		std::cout << "Perimeter " << per << std::endl;
		double thinness = 4.0 * (PI) * ((double)area / (double)(per * per));
		std::cout << "Thinness ratio " << thinness << std::endl;
		double r = (double)(c_max - c_min + 1) / (double)(r_max - r_min + 1);
		std::cout << "Aspect ratio: " << r << std::endl;
		Point p, p1, p2, p3, p4;
		p.x = cs;
		p.y = rs;
		p1.x = p.x + 10;
		p1.y = p.y;
		p2.x = p.x - 10;
		p2.y = p.y;
		p3.x = p.x;
		p3.y = p.y + 10;
		p4.x = p.x;
		p4.y = p.y - 10;
		line(test, p2, p1, col);
		line(test, p3, p4, col);

		double tw0fi = atan2(2 * top_val, (bottom_left - bottom_right));
		tw0fi /= 2;
		double tang = tan(tw0fi);
		int dx = 20;
		int dy = dx * tang;

		Point p11, p21;
		p11.x = cs - dx;
		p11.y = rs - dy;
		p21.x = cs + dx;
		p21.y = rs + dy;
		line(test, p11, p21, col);
		//waitKey(1000);
		imshow("Contour", test);
		Mat projection = Mat(src->rows, src->cols, CV_8UC3);
		for (int i = 0; i < src->rows; i++) {
			int j1 = 0;
			for (int j = 0; j < src->cols; j++) {
				if (src->at<cv::Vec3b>(i, j) == col) {
					projection.at<cv::Vec3b>(i, j1) = col;
					if (isInside(projection, i, j1 + 1))j1++;
				}
			}
		}
		for (int j = 0; j < src->cols; j++) {
			int i1 = src->rows - 1;
			for (int i = 0; i < src->rows; i++) {
				if (src->at<cv::Vec3b>(i, j) == col) {
					projection.at<cv::Vec3b>(i1, j) = col;
					if (isInside(projection, i1 - 1, j))i1--;
				}
			}
		}
		imshow("Projection", projection);
	}
}

void computeGradient(const Mat& src, Mat& gradX, Mat& gradY) {
	Sobel(src, gradX, CV_32F, 1, 0);
	Sobel(src, gradY, CV_32F, 0, 1);
}

void computeStructureTensor(const Mat& gradX, const Mat& gradY, Mat& Sxx, Mat& Sxy, Mat& Syy) {
	Sxx = gradX.mul(gradX);
	Syy = gradY.mul(gradY);
	Sxy = gradX.mul(gradY);
}

void computeHarrisResponse(const Mat& Sxx, const Mat& Sxy, const Mat& Syy, Mat& harris) {
	Mat det = Sxx.mul(Syy) - Sxy.mul(Sxy);
	Mat trace = Sxx + Syy;
	harris = det - 0.04 * trace.mul(trace);
}

void thresholdCorners(const Mat& harris, double thresh, Mat& corners) {
	corners = Mat::zeros(harris.size(), CV_8UC1);
	for (int i = 0; i < harris.rows; ++i) {
		for (int j = 0; j < harris.cols; ++j) {
			if (harris.at<float>(i, j) > thresh) {
				corners.at<uchar>(i, j) = 255;
			}
		}
	}
}

int thresh = 170;
int thresh_offset = 35;
int k = 10;
int k_scaled = 10;
int optionRead;

void on_trackbar(int, void*);
Mat test2(Mat src);

Mat getSrc(int option) {
	switch (option)
	{
	case 1: {
		Mat src = imread("C:/Users/robot/Downloads/Chess_Board.svg.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 2: {
		Mat src = imread("Images/ex1.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 3: {
		Mat src = imread("Images/ex2.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 4: {
		Mat src = imread("Images/ex3.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 5: {
		Mat src = imread("Images/ex4.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 6: {
		Mat src = imread("Images/ex5.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 7: {
		Mat src = imread("Images/ex6.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 8: {
		Mat src = imread("Images/ex7.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 9: {
		Mat src = imread("Images/chessboard.jpg", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 11: {
		Mat src = imread("Images/building.jpg", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	case 10: {
		Mat src = imread("Images/building1.png", IMREAD_GRAYSCALE);
		return src;
		break;
	}
	default:
		break;
	}
}

void on_trackbar(int, void*)
{
	thresh = thresh_offset + 185;
	Mat src = getSrc(optionRead);
	Mat harris_image = test2(src);
	imshow("Harris Corner Detector", harris_image);
}

Mat test2(Mat src) {
	int blockSize = 2;
	int apertureSize = 3;
	double k1 = k / 100.0;

	//Sobel Operator Kernels
	Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat kernel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	//convolution
	Mat dx, dy;
	filter2D(src, dx, CV_32F, kernel_x);
	filter2D(src, dy, CV_32F, kernel_y);

	//partial derivatives
	Mat dx2 = dx.mul(dx);  // dx^2
	Mat dy2 = dy.mul(dy);  // dy^2
	Mat dxy = dx.mul(dy);  // dx*dy

	//apply blur
	Mat dx2_blur, dy2_blur, dxy_blur;

	Mat kernel = Mat::ones(blockSize, blockSize, CV_32F) / (float)(blockSize * blockSize);

	filter2D(dx2, dx2_blur, dx2.depth(), kernel);
	filter2D(dy2, dy2_blur, dy2.depth(), kernel);
	filter2D(dxy, dxy_blur, dxy.depth(), kernel);

	//response
	Mat det = dx2_blur.mul(dy2_blur) - dxy_blur.mul(dxy_blur);
	Mat trace = dx2_blur + dy2_blur;
	Mat harris_response = det - k1 * trace.mul(trace);

	//Threshold
	Mat harris_norm, harris_scaled;
	normalize(harris_response, harris_norm, 0, 255, NORM_MINMAX, CV_32FC1);
	convertScaleAbs(harris_norm, harris_scaled);

	for (int i = 1; i < harris_norm.rows - 1; i++)
	{
		for (int j = 1; j < harris_norm.cols - 1; j++)
		{
			float center = harris_norm.at<float>(i, j);
			if (center <= harris_norm.at<float>(i - 1, j - 1) || center <= harris_norm.at<float>(i - 1, j) ||
				center <= harris_norm.at<float>(i - 1, j + 1) || center <= harris_norm.at<float>(i, j - 1) ||
				center <= harris_norm.at<float>(i, j + 1) || center <= harris_norm.at<float>(i + 1, j - 1) ||
				center <= harris_norm.at<float>(i + 1, j) || center <= harris_norm.at<float>(i + 1, j + 1))
				harris_norm.at<float>(i, j) = 0;
			
		}
	}

	//circle around corners
	for (int i = 0; i < harris_norm.rows; i++)
	{
		for (int j = 0; j < harris_norm.cols; j++)
		{
			if ((int)harris_norm.at<float>(i, j) > thresh)
			{
				circle(harris_scaled, Point(j, i), 2, Scalar(255, 255, 0), 2, 8, 0);
			}
		}
	}

	return harris_scaled;
}


void test() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname);

	imshow("img", img);
	waitKey(0);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	gray.convertTo(gray, CV_32F);
	Mat dst;
	cornerHarris(gray, dst, 2, 3, 0.08);

	dilate(dst, dst, Mat());

	double minVal, maxVal;
	minMaxLoc(dst, &minVal, &maxVal);

	Mat result = img.clone();
	result.setTo(Vec3b(0, 0, 255), dst > 0.01 * maxVal);

	imshow("dst", result);
	waitKey(0);
}

void proj() {
	Mat image;
	char fname[MAX_PATH];
	openFileDlg(fname);
	image = imread(fname, IMREAD_GRAYSCALE);

	/*Mat gradX, gradY, Sxx, Sxy, Syy, harris, corners;
	computeGradient(image, gradX, gradY);
	computeStructureTensor(gradX, gradY, Sxx, Sxy, Syy);
	computeHarrisResponse(Sxx, Sxy, Syy, harris);*/
	std::cout << "Computing gradients..." << std::endl;
	Mat gradX, gradY;
	computeGradient(image, gradX, gradY);
	imshow("Gradient X", gradX);
	imshow("Gradient Y", gradY);
	waitKey(0);

	std::cout << "Computing structure tensor..." << std::endl;
	Mat Sxx, Sxy, Syy;
	computeStructureTensor(gradX, gradY, Sxx, Sxy, Syy);
	imshow("Sxx", Sxx);
	imshow("Sxy", Sxy);
	imshow("Syy", Syy);
	waitKey(0);

	std::cout << "Computing Harris response..." << std::endl;
	Mat harris;
	computeHarrisResponse(Sxx, Sxy, Syy, harris);
	imshow("Harris Response", harris);
	waitKey(0);

	std::cout << "Thresholding corners..." << std::endl;
	double thresh = 10000; 
	Mat corners;
	thresholdCorners(harris, thresh, corners);
	imshow("Corners", corners);
	waitKey(0);

	//double thresh = 0.01 * 255 * 255; 
	thresholdCorners(harris, thresh, corners);

	imshow("Original Image", image);
	imshow("Harris Corners", corners);
	waitKey(0);
}

void lab4()
{
	Mat src;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", CallBackFuncLab4, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);
	std::cout << "Choose a picture by number\n ";
	std::cin >> optionRead;
	imshow("original", getSrc(optionRead));
	waitKey(0);
	namedWindow("Harris Corner Detector", WINDOW_AUTOSIZE);

	createTrackbar("Thresh-185", "Harris Corner Detector", &thresh_offset, 255 - thresh, on_trackbar);
	createTrackbar("k * 100", "Harris Corner Detector", &k, 20, on_trackbar);
	on_trackbar(0, 0);
	waitKey(0);

	return 0;
	
}