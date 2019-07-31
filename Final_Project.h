#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QTimer>
#include <QMessageBox>
#include <qdebug.h>
#include <qprogressbar.h>
#include "ui_Final_Project.h"

//Generic include
#include <windows.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

//OpenCV include
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
using namespace cv;

//OpenCL include
#include "CL/cl.hpp"

//Global variable
//SAD
const int mSize = 2;
//ASW
const int mSize_ASW = 7; //20
const float SIGMA_COLOR = 5.0f;
const float SIGMA_SPATIAL = mSize * 1.1f;
const float truncation = 15.0f;
//BP
const int HIER_LEVELS = 4;
const int HIER_ITER = 10;
//SGM
const float PENALTY1 = 0.2f;
const float PENALTY2 = 50.0f;
#define MMAX(a,b) (((a)>(b))?(a):(b))
#define MMIN(a,b) (((a)<(b))?(a):(b))

class Final_Project : public QMainWindow
{
	Q_OBJECT
		QTimer _timer;

public:
	Final_Project(QWidget *parent = Q_NULLPTR);

private:
	Ui::Final_ProjectClass ui;

	//Stereo parameter
	int dispRange = 64;
	int dispRange_video = 32;

	//QT variables
	QImage qStereoLeft, qStereoRight, QStereoDepth;
	QImage qVideoStream;
	QTimer *qTimer;

	//Mat to QTIMage
	QImage MatToQTImage(Mat mat);

	//OpenCV variables
	Mat left_image;
	Mat right_image;
	Mat left_rectified, right_rectified;
	Mat depth_image;
	Mat left_gray;
	Mat right_gray;
	Mat frame, frame_gray;
	Mat frames;
	//Mat dummy = Mat(1, 1, CV_32FC1, cvScalar(0));
	VideoCapture cap;
	CvCapture *cvCapture;
	VideoCapture webcam;
	float fps = 1000 / 25;
	Mat dummy;

	//OpenCL variables
	cl_int err = CL_SUCCESS;
	vector<cl::Platform> platforms;
	cl::Context context;
	std::vector<cl::Device> devices;
	cl::CommandQueue queue;
	std::string platformVendor;
	//OpenCL kernels
	cl::Kernel Kernel_Rectification, Kernel_Rectification_VGA, Kernel_Rectification_back;
	cl::Kernel Kernel_Gaussian, Kernel_Sobel;
	cl::Kernel Kernel_SAD, Kernel_ASW;
	cl::Kernel Kernel_WTA;

	cl::Kernel Kernel_EvaluatePath;
	cl::Kernel Kernel_IterateXPos;
	cl::Kernel Kernel_IterateYPos;
	cl::Kernel Kernel_IterateXNeg;
	cl::Kernel Kernel_IterateYNeg;
	cl::Kernel Kernel_SumCost;

	cl::Kernel Kernel_InitHierCost;
	cl::Kernel Kernel_HierBP;
	cl::Kernel Kernel_HierBP_Local;
	cl::Kernel Kernel_UpdateCostLayer;
	cl::Kernel Kernel_FinalBP;
	//OpenCL buffers
	cl::Buffer bufferLeft, bufferRight, bufferStereo;
	cl::Buffer bufferGaussianLeft, bufferGaussianRight;
	cl::Buffer bufferSobelLeft, bufferSobelRight;
	cl::Buffer bufferDataCost, bufferSGMCost, bufferSGMAccumulate;
	cl::Buffer bufferDepth, bufferDepthWarp;
	cl::Buffer **bufferHierCost;
	cl::Buffer m_u, m_l, m_d, m_r;
	cl::Buffer temp_m_u, temp_m_l, temp_m_d, temp_m_r;
	cl::Buffer bufferZero;
	//OpenCL events
	cl::Event gaussian, sobel, rectification, rectification_back;
	cl::Event sad, asw, wta;
	cl::Event xPos, yPos, xNeg, yNeg, sumCost, sumCost2;
	cl::Event initHierCost, hierBP, updateCostLayer, finalBP;
	//Generic variables
	int width;
	int height;
	int stereoWidth;
	int totalPixels;
	int totalCost;
	bool firstFrame = true;
	bool showRectified = false;
	bool showStereoFrame = true;
	bool imageLoaded = false;
	int cost_matching = 2; // [1] SAD  [2] ASW

	int *disp;
	float **left;
	float **right;
	float ***datacost;
	float* zeros;
	float* zeros_dispRange;
	uint8_t* deviceToHost;
	float_t* deviceToHost_float;

	float **Gfiltered_imgL;
	float **Gfiltered_imgR;
	float **Sfiltered_imgL;
	float **Sfiltered_imgR;

	//Timer variable
	/*Variable for calculating processing time*/
	LARGE_INTEGER	frequencies;
	LARGE_INTEGER	start;
	LARGE_INTEGER	stop;
	double			timespent = 0;

	private slots:
	void loadImage();
	void SAD_CPU();
	void ASW_CPU();
	void initializeArray();
	int  initOpenCL();
	void initOpenCLBuffer();
	void initOpenCLBufferVideo();
	void SAD_GPU();
	void ASW_GPU();
	void displayVideo();
	void captureFrame();
	void iterateDirection(int dirx, int diry, int nx, int ny, int disp_range, float SMOOTH_PENALTY, float EDGE_PENALTY);
	

	void filter_Gaussian(float **inputL, float **inputR, float **outputL, float **outputR);
	void filter_Sobel(float **inputL, float **inputR, float **outputL, float **outputR);

};
