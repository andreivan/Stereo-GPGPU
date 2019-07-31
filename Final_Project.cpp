/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "Final_Project.h"

// PADDING Y
int index_y(int y, int height)
{
	if (0 <= y && y < height)
		return y;
	else if (y < 0)
		return 0;
	else
		return height - 1;
}
// PADDING X
int index_x(int x, int width)
{
	if (0 <= x && x < width)
		return x;
	else if (x < 0)
		return 0;
	else
		return width - 1;
}

Final_Project::Final_Project(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	move(QPoint(200, 200));
	cap.open(0);

	//Disable maximize and minimize window
	setWindowFlags(Qt::Window | Qt::WindowMinimizeButtonHint | Qt::WindowCloseButtonHint);
	QFont _Font("Arial", 9);
	QApplication::setFont(_Font);
	setWindowTitle("Stereo Matching GPU");
	//Add icon to button
	ui.load_image_button->setIcon(QIcon("Icons/load_image.png"));
	ui.load_image_button->setIconSize(QSize(64, 64));
	ui.load_image_button->setStyleSheet("QPushButton{background: transparent;}");

	ui.video_button->setIcon(QIcon("Icons/video-camera.png"));
	ui.video_button->setIconSize(QSize(64, 64));
	ui.video_button->setStyleSheet("QPushButton{background: transparent;}");

	//Signal and slot
	connect(ui.load_image_button, SIGNAL(clicked()), this, SLOT(loadImage()));
	connect(ui.SAD_CPU_button, SIGNAL(clicked()), this, SLOT(SAD_CPU()));
	connect(ui.ASW_CPU_button, SIGNAL(clicked()), this, SLOT(ASW_CPU()));
	connect(ui.SAD_GPU_button, SIGNAL(clicked()), this, SLOT(SAD_GPU()));
	connect(ui.ASW_GPU_button, SIGNAL(clicked()), this, SLOT(ASW_GPU()));
	//connect(qTimer, SIGNAL(timeout()), this, SLOT(display_video()));
	connect(ui.video_button, SIGNAL(clicked()), this, SLOT(captureFrame()));

	//Initialization QT
	ui.stereo_left->setScaledContents(true);
	ui.stereo_right->setScaledContents(true);
	ui.stereo_depth_CPU->setScaledContents(true);
	ui.stereo_depth_GPU->setScaledContents(true);

}

//Convert Mat(OpenCV) to QTImage
QImage Final_Project::MatToQTImage(Mat mat)
{
	if (mat.channels() == 1)
		return QImage((uchar*)mat.data, (int)mat.cols, (int)mat.rows, (int)mat.step, QImage::Format_Indexed8);
	else if (mat.channels() == 3)
	{
		cvtColor(mat, mat, COLOR_BGR2RGB);
		return QImage((uchar*)mat.data, (int)mat.cols, (int)mat.rows, (int)mat.step, QImage::Format_RGB888);
	}
	else
		cout << "Image is not 1 channel or 3 channels";

	return QImage();
}
void Final_Project::initializeArray()
{
	QString cb = ui.combo_box_disparity->currentText();
	dispRange = cb.toInt();
	cvtColor(left_image, left_gray, CV_BGR2GRAY);
	cvtColor(right_image, right_gray, CV_BGR2GRAY);

	for (int i = 0; i < height; i++)
	{
		datacost[i] = new float*[width];
		left[i] = new float[width];
		right[i] = new float[width];
		Gfiltered_imgL[i] = new float[width];
		Gfiltered_imgR[i] = new float[width];
		Sfiltered_imgL[i] = new float[width];
		Sfiltered_imgR[i] = new float[width];
		for (int j = 0; j < width; j++)
		{
			datacost[i][j] = new float[dispRange];
			for (int d = 0; d < dispRange; d++)
			{
				datacost[i][j][d] = 0;
			}
			left[i][j] = left_gray.at<uchar>(i, j);
			right[i][j] = right_gray.at<uchar>(i, j);
		}
	}


}
void Final_Project::filter_Gaussian(float **inputL, float **inputR, float** outputL, float** outputR)
{
	const float gaussian_kernel[25] = { 0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f,
		0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
		0.000922f, 0.107391f, 0.466066f, 0.107391f, 0.000922f,
		0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
		0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = 0;
			float resultL_x = 0.0f;
			float resultR_x = 0.0f;
			for (int m = -2; m <= 2; m++)
			{
				for (int n = -2; n <= 2; n++)
				{
					int yy = index_y(i + m, height);
					int xx = index_x(j + n, width);
					resultL_x += gaussian_kernel[index] * inputL[yy][xx];
					resultR_x += gaussian_kernel[index] * inputR[yy][xx];
					index++;
				}
			}
			outputL[i][j] = resultL_x;
			outputR[i][j] = resultR_x;
		}
	}
	//return outputL, outputR;
}
void Final_Project::filter_Sobel(float **inputL, float **inputR, float** outputL, float** outputR)
{
	const float sobel_kernelY[25] = { -0.41666f,  -0.03333f,  0.0f,  0.03333f,  0.41666f,
		-0.06666f,  -0.08333f,  0.0f,  0.08333f,  0.06666f,
		-0.08333f,  -0.16666f,  0.0f,  0.16666f,  0.08333f,
		-0.06666f,  -0.08333f,  0.0f,  0.08333f,  0.06666f,
		-0.41666f,  -0.03333f,  0.0f,  0.03333f,  0.41666f };

	const float sobel_kernelX[25] = { -0.41666f, -0.06666f,  -0.08333f, -0.06666f,  -0.41666f,
		-0.03333f, -0.08333f,  -0.16666f, -0.08333f,  -0.03333f,
		0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
		0.03333f, 0.08333f,  0.16666f, 0.08333f,  0.03333f,
		0.41666f, 0.06666f,  0.08333f, 0.06666f,  0.41666f };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = 0;
			float resultL_x = 0.0f;
			float resultR_x = 0.0f;
			float resultL_y = 0.0f;
			float resultR_y = 0.0f;
			for (int m = -2; m <= 2; m++)
			{
				for (int n = -2; n <= 2; n++)
				{
					int yy = index_y(i + m, height);
					int xx = index_x(j + n, width);
					resultL_x += sobel_kernelX[index] * inputL[yy][xx];
					resultR_x += sobel_kernelX[index] * inputR[yy][xx];
					resultL_y += sobel_kernelY[index] * inputL[yy][xx];
					resultR_y += sobel_kernelY[index] * inputR[yy][xx];
					index++;
				}
			}
			outputL[i][j] = sqrt(pow(resultL_x, 2) + pow(resultL_y, 2));
			outputR[i][j] = sqrt(pow(resultR_x, 2) + pow(resultR_y, 2));
		}
	}
	//return outputL, outputR;

}
void Final_Project::loadImage()
{

	//Open window left
	//QString path_f = QFileDialog::getOpenFileName(this, tr("file selection"), ".", tr("Image Files(*.jpg *.png *.tif)"));
	//if (!path_f.isEmpty()) {
	//	left_image = imread(path_f.toStdString());
	//	qStereoLeft = MatToQTImage(left_image);
	//	ui.stereo_left->setPixmap(QPixmap::fromImage(qStereoLeft));
	//}
	//else
	//{
	//	QMessageBox::warning(this, tr("Error"), "Failed to load file!");
	//	ui.stereo_left->clear();
	//	ui.stereo_right->clear();
	//	ui.stereo_depth_CPU->clear();
	//	ui.stereo_depth_GPU->clear();
	//	return;

	//}
	//Mat temp;
	//applyColorMap(left_image*1.00, temp, COLORMAP_JET);
	//imwrite("convert1.png", temp);
	////Open window right
	//path_f = QFileDialog::getOpenFileName(this, tr("file selection"), ".", tr("Image Files(*.jpg *.png *.tif)"));
	//if (!path_f.isEmpty()) {
	//	right_image = imread(path_f.toStdString());
	//	qStereoRight = MatToQTImage(right_image);
	//	ui.stereo_right->setPixmap(QPixmap::fromImage(qStereoRight));
	//}
	//else
	//{
	//	QMessageBox::warning(this, tr("Error"), "Failed to load file!");
	//	ui.stereo_left->clear();
	//	ui.stereo_right->clear();
	//	ui.stereo_depth_CPU->clear();
	//	ui.stereo_depth_GPU->clear();
	//	return;
	//}


	//--For fast image loading in debugging or testing algorithms--
	left_image = imread("Images/USB2-l.png");
	qStereoLeft = MatToQTImage(left_image);
	ui.stereo_left->setPixmap(QPixmap::fromImage(qStereoLeft));
	right_image = imread("Images/USB2-r.png");
	qStereoRight = MatToQTImage(right_image);
	ui.stereo_right->setPixmap(QPixmap::fromImage(qStereoRight));

	//frames = imread("Images/USB2.jpg", 0);

	if (left_image.cols == right_image.cols && left_image.rows == right_image.rows)
	{
		width = left_image.cols;
		height = left_image.rows;
		//Automatically resize the image size to be multiple of 16 for better parallelization
		width = int(width / 16) * 16;
		height = int(height / 16) * 16;
		cv::resize(left_image, left_image, Size(width, height), 0, 0, INTER_CUBIC);
		cv::resize(right_image, right_image, Size(width, height), 0, 0, INTER_CUBIC);
		imageLoaded = true;
	}
	else if (left_image.empty() || right_image.empty())
	{
		QMessageBox::information(0, QString("ERROR"), QString("Image is not loaded"), QMessageBox::Ok);
		imageLoaded = false;
		ui.stereo_left->clear();
		ui.stereo_right->clear();
		ui.stereo_depth_CPU->clear();
		ui.stereo_depth_GPU->clear();
		return;
	}
	else
	{
		QMessageBox::information(0, QString("ERROR"), QString("Image size different"), QMessageBox::Ok);
		imageLoaded = false;
		ui.stereo_left->clear();
		ui.stereo_right->clear();
		ui.stereo_depth_CPU->clear();
		ui.stereo_depth_GPU->clear();
		return;
	}

	ui.stereo_depth_GPU->clear();
	ui.stereo_depth_CPU->clear();

	QString cb = ui.combo_box_disparity->currentText();
	dispRange = cb.toInt();

	//Initialize array based on image width and height
	totalPixels = width * height;
	totalCost = totalPixels * dispRange;
	zeros = new float[totalCost];
	zeros_dispRange = new float[dispRange];
	memset(zeros, 0.0f, sizeof(float)*totalCost);
	memset(zeros_dispRange, 0.0f, sizeof(float)*dispRange);

	disp = new int[totalPixels];
	left = new float*[height];
	right = new float*[height];
	Gfiltered_imgL = new float*[height];
	Gfiltered_imgR = new float*[height];
	Sfiltered_imgL = new float*[height];
	Sfiltered_imgR = new float*[height];
	datacost = new float**[height];
	deviceToHost = new uint8_t[totalPixels]();

	depth_image = Mat(height, width, CV_8UC1, cvScalar(0));

	//Initialize image array
	initializeArray();

	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);

	//Initialization OpenCL
	int error = initOpenCL();
	if (error != 1) {
		QMessageBox::information(0, QString("ERROR"), QString("Failed to initialize OpenCL").arg(error), QMessageBox::Ok);
	}
	else
	{
		initOpenCLBuffer();
		QueryPerformanceCounter(&stop);
		timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;
		QMessageBox::information(0, QString("NOTICE"), QString("Successfully to initialize OpenCL \nin %1 seconds").arg(timespent), QMessageBox::Ok);
	}

}

//New SGM not finished yet
void evaluate_path(const float *prior, const float *local,
	float path_intensity_gradient, float *curr_cost,
	int nx, int ny, int disp_range)
{
	memcpy(curr_cost, local, sizeof(int)*disp_range);

	for (int d = 0; d < disp_range; d++) {
		int e_smooth = std::numeric_limits<int>::max();
		for (int d_p = 0; d_p < disp_range; d_p++) {
			if (d_p - d == 0) {
				// No penality
				e_smooth = MMIN(e_smooth, prior[d_p]);
			}
			else if (abs(d_p - d) == 1) {
				// Small penality
				e_smooth = MMIN(e_smooth, prior[d_p] + PENALTY1);
			}
			else {
				// Large penality
				e_smooth =
					MMIN(e_smooth, prior[d_p] +
						MMAX(PENALTY1,
							path_intensity_gradient ? PENALTY2 / path_intensity_gradient : PENALTY2));
			}
		}
		curr_cost[d] += e_smooth;
	}

	int min = std::numeric_limits<int>::max();
	for (int d = 0; d < disp_range; d++) {
		if (prior[d]<min) min = prior[d];
	}
	for (int d = 0; d < disp_range; d++) {
		curr_cost[d] -= min;
	}
}

void iterate_direction_dirxpos(int dirx, float **left_image,
	float ***costs, float ***accumulated_costs,
	int nx, int ny, int disp_range)
{
	int WIDTH = nx;
	int HEIGHT = ny;

	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
			if (i == 0) {
				for (int d = 0; d < disp_range; d++) {
					accumulated_costs[j][0][d] += costs[j][0][d];
				}
			}
			else
			{
				evaluate_path(&accumulated_costs[j][i-dirx][0],
					&costs[j][i][0],
					abs(left_image[j][i] - left_image[j][i-dirx]),
					&accumulated_costs[j][i][0], nx, ny, disp_range);
			}
		}
	}
}

void iterate_direction_dirypos(int diry, float **left_image,
	float ***costs, float ***accumulated_costs,
	int nx, int ny, int disp_range)
{
	int WIDTH = nx;
	int HEIGHT = ny;

	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < HEIGHT; j++) {
			if (j == 0) {
				for (int d = 0; d < disp_range; d++) {
					accumulated_costs[0][i][d] += costs[0][i][d];
				}
			}
			else {
				evaluate_path(&accumulated_costs[j - diry][i][0],
					&costs[j][i][0],
					abs(left_image[j][i] - left_image[j - diry][i]),
					&accumulated_costs[j][i][0], nx, ny, disp_range);
			}
		}
	}
}

void iterate_direction_dirxneg(int dirx, float **left_image,
	float ***costs, float ***accumulated_costs,
	int nx, int ny, int disp_range)
{
	int WIDTH = nx;
	int HEIGHT = ny;

	for (int j = 0; j < HEIGHT; j++) {
		for (int i = WIDTH - 1; i >= 0; i--) {
			if (i == WIDTH - 1) {
				for (int d = 0; d < disp_range; d++) {
					accumulated_costs[j][WIDTH - 1][d] += costs[j][WIDTH - 1][d];
				}
			}
			else {		
				evaluate_path(&accumulated_costs[j][i - dirx][0],
					&costs[j][i][0],
					abs(left_image[j][i] - left_image[j][i-dirx]),
					&accumulated_costs[j][i][0], nx, ny, disp_range);
			}
		}
	}
}

void iterate_direction_diryneg(int diry, float **left_image,
	float ***costs, float ***accumulated_costs,
	int nx, int ny, int disp_range)
{
	int WIDTH = nx;
	int HEIGHT = ny;

	for (int i = 0; i < WIDTH; i++) {
		for (int j = HEIGHT - 1; j >= 0; j--) {
			if (j == HEIGHT - 1) {
				for (int d = 0; d < disp_range; d++) {
					accumulated_costs[HEIGHT - 1][i][d] += costs[HEIGHT - 1][i][d];
				}
			}
			else {
				
				evaluate_path(&accumulated_costs[j - diry][i][0],
					&costs[j][i][0],
					abs(left_image[j][i] - left_image[j - diry][i]),
					&accumulated_costs[j][i][0], nx, ny, disp_range);
			}
		}
	}
}

void iterate_direction(int dirx, int diry, float **left_image,
	float ***costs, float ***accumulated_costs,
	int nx, int ny, int disp_range)
{
	// Walk along the edges in a clockwise fashion
	if (dirx > 0)
	{
		// LEFT MOST EDGE
		// Process every pixel along this edge
		iterate_direction_dirxpos(dirx, left_image, costs, accumulated_costs, nx, ny, disp_range);
	}
	else if (diry > 0) {
		// TOP MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the top left most pixel
		iterate_direction_dirypos(diry, left_image, costs, accumulated_costs, nx, ny, disp_range);
	}
	else if (dirx < 0) {
		// RIGHT MOST EDGE
		// Process every pixel along this edge only if diry ==
		// 0. Otherwise skip the top right most pixel
		iterate_direction_dirxneg(dirx, left_image, costs, accumulated_costs, nx, ny, disp_range);
	}
	else if (diry < 0) {
		// BOTTOM MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the bottom left and bottom right pixel
		iterate_direction_diryneg(diry, left_image, costs, accumulated_costs, nx, ny, disp_range);
	}
}

// ADD two cost images 
void inplace_sum_views(float ***im1, float ***im2,
	int nx, int ny, int disp_range)
{
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			for (int d = 0; d < disp_range; d++)
			{
				im1[i][j][d] += im2[i][j][d];
			}
		}
	}
}

float*** SGM(float ** left, float ***datacost, int width, int height, int dispRange)
{
	float *** accumulated_costs = new float**[height];
	float *** dir_accumulated_costs = new float**[height];
	for (int i = 0; i < height; i++)
	{
		accumulated_costs[i] = new float*[width];
		dir_accumulated_costs[i] = new float*[width];
		for (int j = 0; j < width; j++)
		{
			accumulated_costs[i][j] = new float[dispRange];
			dir_accumulated_costs[i][j] = new float[dispRange];
			for (int d = 0; d < dispRange; d++)
			{
				accumulated_costs[i][j][d] = 0;
				dir_accumulated_costs[i][j][d] = 0;
			}
		}
	}

	int dirx = 0, diry = 0;
	for (dirx = -1; dirx < 2; dirx++) 
	{
		if (dirx == 0 && diry == 0) continue;
		//for (int i = 0; i < height; i++)
		//{
		//	for (int j = 0; j < width; j++)
		//	{
		//		for (int d = 0; d < dispRange; d++)
		//		{
		//			dir_accumulated_costs[i][j][d] = 0;
		//		}
		//	}
		//}
		iterate_direction(dirx, diry, left, datacost, dir_accumulated_costs, width, height, dispRange);
		inplace_sum_views(accumulated_costs, dir_accumulated_costs, width, height, dispRange);
	}
	dirx = 0;
	for (diry = -1; diry<2; diry++) {
		if (dirx == 0 && diry == 0) continue;
		//for (int i = 0; i < height; i++)
		//{
		//	for (int j = 0; j < width; j++)
		//	{
		//		for (int d = 0; d < dispRange; d++)
		//		{
		//			dir_accumulated_costs[i][j][d] = 0;
		//		}
		//	}
		//}
		iterate_direction(dirx, diry, left, datacost, dir_accumulated_costs, width, height, dispRange);
		inplace_sum_views(accumulated_costs, dir_accumulated_costs, width, height, dispRange);
	}
	return accumulated_costs;
}


//BP Optimization
void comp_msg(float *m1, float *m2, float *m3, float *data, float *dest, int MAX_DISP, float CONT_COST, float DISCONT_COST)
{
	float acc;
	float prev, cur, tmp;
	float minimum;
	int q;
	//Data cost update initialize
	dest[0] = m1[0] + m2[0] + m3[0] + data[0];
	minimum = dest[0];
	//1 until disparity range Equation(2) with penalty
	for (q = 1; q < MAX_DISP; q++)
	{
		prev = dest[q - 1] + CONT_COST;
		//Data cost update again Equation (2) paper
		cur = m1[q] + m2[q] + m3[q] + data[q];
		tmp = (prev < cur) ? prev : cur;
		dest[q] = tmp;
		//Store the min to minimum
		minimum = (tmp < minimum) ? tmp : minimum;
	}

	minimum += (float)DISCONT_COST;
	dest[MAX_DISP - 1] = (minimum < dest[MAX_DISP - 1]) ? minimum : dest[MAX_DISP - 1];
	acc = dest[MAX_DISP - 1];
	//Inverse direction
	for (q = MAX_DISP - 2; q >= 0; q--)
	{
		prev = dest[q + 1] + CONT_COST;
		prev = (minimum < prev) ? minimum : prev;
		dest[q] = (prev < dest[q]) ? prev : dest[q];
		acc += dest[q];
	}

	acc /= (float)MAX_DISP;
	//Reduce with acc for all cost
	for (q = 0; q < MAX_DISP; q++)
	{
		dest[q] -= acc;
	}

}

void hier_bp(float **datacost, float *m_u, float *m_d, float *m_l, float *m_r, int *dWidth, int *dHeight, int MAX_DISP, int HIER_LEVEL, int HIER_ITER, float CONT_COST, float DISCONT_COST)
{
	float *temp_u, *temp_l, *temp_r, *temp_d;
	float *zero;
	int dist = 1;
	zero = new float[MAX_DISP];
	memset(zero, 0.0f, sizeof(float)*MAX_DISP);

	for (int i = 1; i < HIER_LEVEL; i++)
	{
		dist *= 2; //2^(HIER_LEVEL-1)
	}

	//Top down updating smoothness cost iteratively
	for (int level = HIER_LEVEL - 1; level >= 0; level--)
	{
		std::cout << "LEVEL:" << level << std::endl;
		//Iterative message pass
		for (int iter = 0; iter<HIER_ITER*(level*level + 1); iter++)
		{
			//std::cout << "ITER:" << iter << std::endl;
			for (int i = 0; i<dHeight[level]; i++)
			{
				for (int j = ((i + iter) % 2); j<dWidth[level]; j += 2)
				{
					//Edges True: set zero || False: set pointer 
					temp_u = (i + 1 >= dHeight[level]) ? zero : &m_u[(dWidth[0] * (i + 1)*dist + j * dist)*MAX_DISP];
					temp_l = (j + 1 >= dWidth[level]) ? zero : &m_l[(dWidth[0] * (i)*dist + (j + 1)*dist)*MAX_DISP];
					temp_r = (j - 1 < 0) ? zero : &m_r[(dWidth[0] * (i)*dist + (j - 1)*dist)*MAX_DISP];
					temp_d = (i - 1 < 0) ? zero : &m_d[(dWidth[0] * (i - 1)*dist + j * dist)*MAX_DISP];

					//Compute "belief" message in four directions up, down, right, and left
					comp_msg(temp_u, temp_l, temp_r, &datacost[level][(dWidth[level] * (i)+j)*MAX_DISP], &m_u[(dWidth[0] * (i)*dist + j * dist)*MAX_DISP], MAX_DISP, CONT_COST, DISCONT_COST);
					comp_msg(temp_d, temp_l, temp_r, &datacost[level][(dWidth[level] * (i)+j)*MAX_DISP], &m_d[(dWidth[0] * (i)*dist + j * dist)*MAX_DISP], MAX_DISP, CONT_COST, DISCONT_COST);
					comp_msg(temp_u, temp_d, temp_r, &datacost[level][(dWidth[level] * (i)+j)*MAX_DISP], &m_r[(dWidth[0] * (i)*dist + j * dist)*MAX_DISP], MAX_DISP, CONT_COST, DISCONT_COST);
					comp_msg(temp_u, temp_d, temp_l, &datacost[level][(dWidth[level] * (i)+j)*MAX_DISP], &m_l[(dWidth[0] * (i)*dist + j * dist)*MAX_DISP], MAX_DISP, CONT_COST, DISCONT_COST);
					//std::cout << "INDEX datacost:" << (dWidth[level] * (i)+j)*MAX_DISP << std::endl;
					//std::cout << "INDEX smooth:" << (dWidth[0] * (i)*dist + j * dist)*MAX_DISP << std::endl;
				}
			}
		}

		//Update the smoothness cost for next layer
		if (level != 0)
		{
			for (int i = 0; i<dHeight[level]; i++)
			{
				for (int j = 0; j<dWidth[level]; j++)
				{
					for (int oi = 0; oi<dist; oi++)
					{
						for (int oj = 0; oj<dist; oj++)
						{
							int k = i * dist + oi;
							int l = j * dist + oj;
							if (k >= dHeight[0] || l >= dWidth[0])	break;
							for (int d = 0; d<MAX_DISP; d++)
							{
								m_u[(k*dWidth[0] + l) * MAX_DISP + d] = m_u[(i*dist*dWidth[0] + j * dist) * MAX_DISP + d];
								m_l[(k*dWidth[0] + l) * MAX_DISP + d] = m_l[(i*dist*dWidth[0] + j * dist) * MAX_DISP + d];
								m_d[(k*dWidth[0] + l) * MAX_DISP + d] = m_d[(i*dist*dWidth[0] + j * dist) * MAX_DISP + d];
								m_r[(k*dWidth[0] + l) * MAX_DISP + d] = m_r[(i*dist*dWidth[0] + j * dist) * MAX_DISP + d];
							}
						}
					}
				}
			}
		}
		dist /= 2;
	}



	delete[] zero;
}

void final_bp(float *m_u, float *m_d, float *m_l, float *m_r, float* datacost, int width, int height, int MAX_DISP)
{
	int i;
	int j;
	int k;
	float temp;

	float *zero;
	float *temp_u;
	float *temp_d;
	float *temp_l;
	float *temp_r;
	zero = new float[MAX_DISP];
	memset(zero, 0.0f, sizeof(float)*MAX_DISP);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			//Get smoothness cost[dispRange] and store in temp || edge = zero[dispRange]
			temp_u = (i != height - 1) ? &m_u[((i + 1)*width + j)*MAX_DISP] : zero;
			temp_d = (i != 0) ? &m_d[((i - 1)*width + j)*MAX_DISP] : zero;
			temp_l = (j != width - 1) ? &m_l[(i*width + j + 1)*MAX_DISP] : zero;
			temp_r = (j != 0) ? &m_r[(i*width + j - 1)*MAX_DISP] : zero;
			//Do final belief propagation in the lowest layer (Equation 2) with all 4 neighbors
			for (k = 0; k < MAX_DISP; k++)
			{
				temp = temp_u[k]
					+ temp_d[k]
					+ temp_l[k]
					+ temp_r[k]
					+ datacost[(i*width + j)*MAX_DISP + k];
				//Put the final cost in the datacost
				datacost[(i*width + j)*MAX_DISP + k] = temp;
			}
		}
	}
	delete[] zero;
}

void optimizeBeliefPropagation(float ***datacost, int width, int height, int dispRange, int HIER_ITER, int HIER_LEVEL, float CONT_COST, float DISCONT_COST)
{
	float **cost;
	int *dWidth;
	int *dHeight;
	float *m_u, *m_l, *m_r, *m_d;  //up left right down

	cost = new float*[HIER_LEVEL];
	dWidth = new int[HIER_LEVEL];
	dHeight = new int[HIER_LEVEL];

	for (int i = 0; i<HIER_LEVEL; i++)
	{
		if (i == 0)
		{
			dWidth[i] = width;
			dHeight[i] = height;
		}
		else
		{
			dWidth[i] = dWidth[i - 1] / 2;;
			dHeight[i] = dHeight[i - 1] / 2;
		}
		cost[i] = new float[dWidth[i] * dHeight[i] * dispRange];
	}

	m_u = new float[width*height*dispRange];
	m_l = new float[width*height*dispRange];
	m_d = new float[width*height*dispRange];
	m_r = new float[width*height*dispRange];
	//Initialize smoothness with 0.0f
	memset(m_u, 0.0f, sizeof(float)*width*height*dispRange);
	memset(m_l, 0.0f, sizeof(float)*width*height*dispRange);
	memset(m_d, 0.0f, sizeof(float)*width*height*dispRange);
	memset(m_r, 0.0f, sizeof(float)*width*height*dispRange);

	//Initialize the first layer of hier with cost from stereo matching
	for (int d = 0; d < dispRange; d++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				cost[0][(i*width + j)*dispRange + d] = datacost[i][j][d];
			}
		}
	}

	//Construct the hier cost
	for (int iter = 1; iter<HIER_LEVEL; iter++)
	{
		for (int i = 0; i < dHeight[iter]; i++)
		{
			for (int j = 0; j < dWidth[iter]; j++)
			{
				for (int d = 0; d < dispRange; d++)
				{
					if ((i * 2 + 1) >= dHeight[iter - 1] || (j * 2 + 1) >= dWidth[iter - 1])
					{
						cost[iter][dispRange * (dWidth[iter] * i + j) + d] = 0.0F;
					}
					else
					{
						//Next hier cost is from 4 neighbor cost of prev layer hier
						cost[iter][dispRange * (dWidth[iter] * i + j) + d] = cost[iter - 1][dispRange * (dWidth[iter - 1] * (i * 2) + (j * 2)) + d] +
							cost[iter - 1][dispRange * (dWidth[iter - 1] * (i * 2 + 1) + (j * 2)) + d] +
							cost[iter - 1][dispRange * (dWidth[iter - 1] * (i * 2) + (j * 2 + 1)) + d] +
							cost[iter - 1][dispRange * (dWidth[iter - 1] * (i * 2 + 1) + (j * 2 + 1)) + d];
					}
				}
			}
		}
	}
	//Propagate belief message
	hier_bp(cost, m_u, m_d, m_l, m_r, dWidth, dHeight, dispRange, HIER_LEVEL, HIER_ITER, CONT_COST, DISCONT_COST);
	final_bp(m_u, m_d, m_l, m_r, cost[0], width, height, dispRange);

	for (int d = 0; d < dispRange; d++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				datacost[i][j][d] = cost[0][(i*width + j)*dispRange + d];
			}
		}
	}

	for (int i = 0; i<HIER_LEVEL; i++)
	{
		delete[] cost[i];
	}
	delete[] m_l;
	delete[] m_r;
	delete[] m_d;
	delete[] m_u;
	delete[] cost;
	delete[] dWidth;
	delete[] dHeight;

}

void Final_Project::SAD_CPU()
{
	QString cb = ui.combo_box_optimization->currentText();
	QString cb2 = ui.combo_box_color->currentText();

	if (!imageLoaded)
	{
		QMessageBox::information(0, QString("NOTICE"), QString("Load image first"), QMessageBox::Ok);
		return;
	}
	float DISCONT_COST = 0.4f;
	float CONT_COST = 0.02f;

	QApplication::setOverrideCursor(Qt::WaitCursor);
	//Start timer
	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);
	
	filter_Gaussian(left, right, Gfiltered_imgL, Gfiltered_imgR);
	filter_Sobel(Gfiltered_imgL, Gfiltered_imgR, Sfiltered_imgL, Sfiltered_imgR);

	//Get disparity
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < dispRange; d++)
			{
				if (j - d >= 0)
				{
					float diff = 0;
					for (int m = -mSize; m <= mSize; m++)
					{
						for (int n = -mSize; n <= mSize; n++)
						{
							int x1 = j + n;
							int x2 = j + n - d;
							int y = i + m;
							x1 = index_x(x1, width);
							x2 = index_x(x2, width);
							y = index_y(y, height);
							// Absolute difference
							float _diff = Sfiltered_imgL[y][x1] - Sfiltered_imgR[y][x2];
							if (_diff < 0) {
								_diff = _diff * -1;
							}
							diff += _diff;	
						}
					}
					datacost[i][j][d] = diff / 255.0f;
				}
				else
				{
					datacost[i][j][d] = 1000.0f;
				}
			}
		}
	}

	if (cb.toStdString() == "BP")
	{
		optimizeBeliefPropagation(datacost, width, height, dispRange, HIER_ITER, HIER_LEVELS, CONT_COST, DISCONT_COST);
		
	}
	else if (cb.toStdString() == "SGM")
	{
		datacost = SGM(Sfiltered_imgL, datacost, width, height, dispRange);
	}

	//WTA
	float scale = 256 / dispRange;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float minCost = FLT_MAX;
			int minIndex = 0;
			for (int d = 0; d < dispRange; d++)
			{
				if (minCost > datacost[i][j][d])
				{
					minCost = datacost[i][j][d];
					minIndex = d;
				}
			}
			disp[i*width + j] = minIndex;
			depth_image.at<uchar>(i, j) = disp[i*width + j] * scale;
		}
	}

	//Stop timer
	QueryPerformanceCounter(&stop);
	timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;

	QMessageBox::information(0, QString("NOTICE"), QString("Depthmap from SAD CPU"), QMessageBox::Ok);
	QApplication::restoreOverrideCursor();

	Mat depth_color;
	switch (cb2.toInt())
	{
	case 1:
		applyColorMap(depth_image, depth_color, COLORMAP_BONE);
		break;
	case 2:
		applyColorMap(depth_image, depth_color, COLORMAP_JET);
		break;
	case 3:
		applyColorMap(depth_image, depth_color, COLORMAP_HOT);
		break;
	case 4:
		applyColorMap(depth_image, depth_color, COLORMAP_WINTER);
		break;
	}
	QStereoDepth = MatToQTImage(depth_color);
	ui.stereo_depth_CPU->setPixmap(QPixmap::fromImage(QStereoDepth));
	ui.CPU_time->setText(QString::number(timespent) + " seconds");

}

void Final_Project::ASW_CPU()
{
	QString cb = ui.combo_box_optimization->currentText();
	QString cb2 = ui.combo_box_color->currentText();
	if (!imageLoaded)
	{
		QMessageBox::information(0, QString("NOTICE"), QString("Load image first"), QMessageBox::Ok);
		return;
	}
	float DISCONT_COST = 0.01f;
	float CONT_COST = 0.05f;

	QApplication::setOverrideCursor(Qt::WaitCursor);
	//Start timer
	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);

	filter_Gaussian(left, right, Gfiltered_imgL, Gfiltered_imgR);
	filter_Sobel(Gfiltered_imgL, Gfiltered_imgR, Sfiltered_imgL, Sfiltered_imgR);
	
	//Get disparity
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < dispRange; d++)
			{
				if (j - d >= 0)
				{
					float diff = 0;
					float weight = 0;
					for (int m = -mSize_ASW; m <= mSize_ASW; m++)
					{
						for (int n = -mSize_ASW; n <= mSize_ASW; n++)
						{
							int x1 = j + n;
							int x2 = j + n - d;
							int y = i + m;
							x1 = index_x(x1, width);
							x2 = index_x(x2, width);
							y = index_y(y, height);
							if (x1 >= 0 && y >= 0 && x1 < width && y < height && x2 >= 0 && x2 < width)
							{
								float weightL, weightR;
								float colorDiff, spatialDiff;
								spatialDiff = sqrt((float)m*m + n * n);

								colorDiff = sqrt((float)(Gfiltered_imgL[y][x1] - Gfiltered_imgL[i][j])*(Gfiltered_imgL[y][x1] - Gfiltered_imgL[i][j]));
								weightL = exp(-1.0 * ((colorDiff / SIGMA_COLOR) + (spatialDiff / SIGMA_SPATIAL)));

								colorDiff = sqrt((float)(Gfiltered_imgR[y][x2] - Gfiltered_imgR[i][j - d])*(Gfiltered_imgR[y][x2] - Gfiltered_imgR[i][j - d]));
								weightR = exp(-1.0 * ((colorDiff / SIGMA_COLOR) + (spatialDiff / SIGMA_SPATIAL)));

								float diff_temp = abs(Gfiltered_imgL[y][x1] - Gfiltered_imgR[y][x2]);
								if (diff_temp > truncation)	diff_temp = truncation;
								diff += diff_temp * weightL * weightR / 255.0;
								//diff += abs(Sfiltered_imgL[y][x1] - Sfiltered_imgR[y][x2]) * weightL * weightR;
								weight += weightL * weightR;
							}
						}
					}
					datacost[i][j][d] = diff / weight;
				}
				else
				{
					datacost[i][j][d] = 100000;
				}
			}
		}
	}

	if (cb.toStdString() == "BP")
	{
		optimizeBeliefPropagation(datacost, width, height, dispRange, HIER_ITER, HIER_LEVELS, CONT_COST, DISCONT_COST);
	}
	else if (cb.toStdString() == "SGM")
	{
		datacost = SGM(Sfiltered_imgL, datacost, width, height, dispRange);
	}

	//WTA
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float minCost = FLT_MAX;
			int minIndex = 0;
			for (int d = 0; d < dispRange; d++)
			{
				if (minCost > datacost[i][j][d])
				{
					minCost = datacost[i][j][d];
					minIndex = d;
				}
			}
			disp[i*width + j] = minIndex;
			depth_image.at<uchar>(i, j) = disp[i*width + j] * (256 / dispRange);
		}
	}

	//Stop timer
	QueryPerformanceCounter(&stop);
	timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;

	QMessageBox::information(0, QString("NOTICE"), QString("Depthmap from ASW CPU"), QMessageBox::Ok);
	QApplication::restoreOverrideCursor();
	Mat depth_color;
	switch (cb2.toInt())
	{
	case 1:
		applyColorMap(depth_image, depth_color, COLORMAP_BONE);
		break;
	case 2:
		applyColorMap(depth_image, depth_color, COLORMAP_JET);
		break;
	case 3:
		applyColorMap(depth_image, depth_color, COLORMAP_HOT);
		break;
	case 4:
		applyColorMap(depth_image, depth_color, COLORMAP_WINTER);
		break;
	}
	QStereoDepth = MatToQTImage(depth_color);
	ui.stereo_depth_CPU->setPixmap(QPixmap::fromImage(QStereoDepth));
	ui.CPU_time->setText(QString::number(timespent) + " seconds");
}

int Final_Project::initOpenCL()
{
	//Query platfroms
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cout << "Platform size 0\n";
		return -1;
	}

	// Get the number of platform and information about the platform
	std::cout << "Platform number is: " << platforms.size() << std::endl;


	for (unsigned int i = 0; i < platforms.size(); ++i) {
		platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		std::cout << "Platform is by: " << platformVendor << std::endl;
	}

	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM,(cl_context_properties)(platforms[0])(),0 };

	context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	std::cout << "Device number is: " << devices.size() << std::endl;
	for (unsigned int i = 0; i < devices.size(); ++i) {
		std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	// Generate commane queue
	std::cout << "making command queue for device[0]" << std::endl;
	queue = cl::CommandQueue(context, devices[0], 0, &err);

	ifstream sourceFile_filters("filters.cl");
	string sourceCode_filters(istreambuf_iterator<char>(sourceFile_filters), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_filters(1, make_pair(sourceCode_filters.c_str(), sourceCode_filters.length() + 1));
	cl::Program program_filters = cl::Program(context, source_filters);
	program_filters.build(devices);
	if (program_filters.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -2;

	ifstream sourceFile_recti("rectification.cl");
	string sourceCode_recti(istreambuf_iterator<char>(sourceFile_recti), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_recti(1, make_pair(sourceCode_recti.c_str(), sourceCode_recti.length() + 1));
	cl::Program program_recti = cl::Program(context, source_recti);
	program_recti.build(devices);
	if (program_recti.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -3;

	ifstream sourceFile_SAD("sad_disp.cl");
	string sourceCode_SAD(istreambuf_iterator<char>(sourceFile_SAD), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_SAD(1, make_pair(sourceCode_SAD.c_str(), sourceCode_SAD.length() + 1));
	cl::Program program_SAD = cl::Program(context, source_SAD);
	program_SAD.build(devices);
	if (program_SAD.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -4;

	ifstream sourceFile_ASW("asw_disp.cl");
	string sourceCode_ASW(istreambuf_iterator<char>(sourceFile_ASW), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_ASW(1, make_pair(sourceCode_ASW.c_str(), sourceCode_ASW.length() + 1));
	cl::Program program_ASW = cl::Program(context, source_ASW);
	program_ASW.build(devices);
	if (program_ASW.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -5;

	ifstream sourceFile_WTA("wta.cl");
	string sourceCode_WTA(istreambuf_iterator<char>(sourceFile_WTA), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_WTA(1, make_pair(sourceCode_WTA.c_str(), sourceCode_WTA.length() + 1));
	cl::Program program_WTA = cl::Program(context, source_WTA);
	program_WTA.build(devices);
	if (program_WTA.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -6;

	ifstream sourceFile_SGM("sgm.cl");
	string sourceCode_SGM(istreambuf_iterator<char>(sourceFile_SGM), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_SGM(1, make_pair(sourceCode_SGM.c_str(), sourceCode_SGM.length() + 1));
	cl::Program program_SGM = cl::Program(context, source_SGM);
	program_SGM.build(devices);
	if (program_SGM.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -7;

	ifstream sourceFile_BP("hier_bp.cl");
	string sourceCode_BP(istreambuf_iterator<char>(sourceFile_BP), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source_BP(1, make_pair(sourceCode_BP.c_str(), sourceCode_BP.length() + 1));
	cl::Program program_BP = cl::Program(context, source_BP);
	program_BP.build(devices);
	if (program_BP.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS)
		return -8;
	//Pre-proc
	Kernel_Rectification = cl::Kernel(program_recti, "rectify", &err);
	Kernel_Rectification_VGA = cl::Kernel(program_recti, "rectify_VGA", &err);
	Kernel_Rectification_back = cl::Kernel(program_recti, "rectify_back", &err);
	Kernel_Gaussian = cl::Kernel(program_filters, "gaussian_filter", &err);
	Kernel_Sobel = cl::Kernel(program_filters, "sobel_filter", &err);
	//Local
	Kernel_SAD = cl::Kernel(program_SAD, "SAD_Disparity", &err);
	Kernel_ASW = cl::Kernel(program_ASW, "ASW_Disparity", &err);
	Kernel_WTA = cl::Kernel(program_WTA, "winner_takes_all", &err);
	//SGM
	Kernel_IterateXPos = cl::Kernel(program_SGM, "iterate_direction_dirxpos_ocl", &err);
	Kernel_IterateYPos = cl::Kernel(program_SGM, "iterate_direction_dirypos_ocl", &err);
	Kernel_IterateXNeg = cl::Kernel(program_SGM, "iterate_direction_dirxneg_ocl", &err);
	Kernel_IterateYNeg = cl::Kernel(program_SGM, "iterate_direction_diryneg_ocl", &err);
	Kernel_SumCost = cl::Kernel(program_SGM, "inplace_sum_views_ocl", &err);
	//BP
	Kernel_InitHierCost = cl::Kernel(program_BP, "initializeHierCost", &err);
	Kernel_HierBP = cl::Kernel(program_BP, "hierBP", &err);
	Kernel_UpdateCostLayer = cl::Kernel(program_BP, "updateCostLayer", &err);
	Kernel_FinalBP = cl::Kernel(program_BP, "finalBP", &err);

	if (err != CL_SUCCESS)
		return -9;
	else
		return 1;
}

void Final_Project::initOpenCLBuffer()
{
	bufferLeft = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * sizeof(cl_uchar), left_gray.data, NULL);
	bufferRight = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * sizeof(cl_uchar), right_gray.data, NULL);
	bufferGaussianLeft = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferGaussianRight = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferSobelLeft = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferSobelRight = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	//bufferDataCost = cl::Buffer(context, CL_MEM_READ_WRITE, (totalCost) * sizeof(cl_float));
	bufferSGMCost = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	bufferSGMAccumulate = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	bufferDepth = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));

	QString cb = ui.combo_box_disparity->currentText();
	dispRange = cb.toInt();

	//Data cost is the first hier layer [0]
	bufferHierCost =  new cl::Buffer*[HIER_LEVELS]; 
	int scale = 1;
	for (int level = 0; level < HIER_LEVELS; level++)
	{	
		bufferHierCost[level] = new cl::Buffer(context, CL_MEM_READ_WRITE, (width / scale) * (height / scale) * dispRange * sizeof(cl_float));
		scale *= 2;
	}

	//Smoothnes cost
	m_u = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_l = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_r = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);

	temp_m_u = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_l = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_r = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);

	bufferZero = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dispRange * sizeof(cl_float), zeros_dispRange, NULL);
	//For checking OpenCL
	//deviceToHost_float = new float[(width / 2) * (height / 2) * dispRange]();
}

void Final_Project::initOpenCLBufferVideo()
{
	totalCost = totalPixels * dispRange_video;
	zeros = new float[totalCost];
	zeros_dispRange = new float[dispRange_video];
	memset(zeros, 0.0f, sizeof(float)*totalCost);
	memset(zeros_dispRange, 0.0f, sizeof(float)*dispRange_video);

	depth_image = Mat(height, width, CV_8UC1, cvScalar(0));
	left_rectified = Mat(height, width, CV_8UC1, cvScalar(0));
	right_rectified = Mat(height, width, CV_8UC1, cvScalar(0));
	left_image = Mat(height, width, CV_8UC1, cvScalar(0));
	right_image = Mat(height, width, CV_8UC1, cvScalar(0));
	deviceToHost = new uint8_t[totalPixels]();
	bufferLeft = cl::Buffer(context, CL_MEM_READ_WRITE, totalPixels * sizeof(cl_uchar));
	bufferRight = cl::Buffer(context, CL_MEM_READ_WRITE, totalPixels * sizeof(cl_uchar));
	bufferGaussianLeft = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferGaussianRight = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferSobelLeft = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferSobelRight = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	//bufferDataCost = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * (dispRange_video) * sizeof(cl_float));
	bufferSGMCost = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	bufferSGMAccumulate = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	bufferHierCost = new cl::Buffer*[HIER_LEVELS];
	int scale = 1;
	for (int level = 0; level < HIER_LEVELS; level++)
	{
		bufferHierCost[level] = new cl::Buffer(context, CL_MEM_READ_WRITE, (width / scale) * (height / scale) * dispRange_video * sizeof(cl_float));
		scale *= 2;
	}
	bufferDepth = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));
	bufferDepthWarp = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_uchar));

	//Smoothnes cost
	m_u = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_l = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	m_r = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);

	temp_m_u = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_l = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	temp_m_r = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalCost * sizeof(cl_float), zeros, NULL);
	bufferZero = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dispRange_video * sizeof(cl_float), zeros_dispRange, NULL);
}

void Final_Project::iterateDirection(int dirx, int diry, int nx, int ny, int disp_range, float SMOOTH_PENALTY, float EDGE_PENALTY)
{
	// Walk along the edges in a clockwise fashion
	if (dirx > 0) 
	{
		Kernel_IterateXPos.setArg(0, dirx);
		Kernel_IterateXPos.setArg(1, bufferLeft);
		Kernel_IterateXPos.setArg(2, *bufferHierCost[0]);
		Kernel_IterateXPos.setArg(3, bufferSGMCost);
		Kernel_IterateXPos.setArg(4, nx);
		Kernel_IterateXPos.setArg(5, ny);
		Kernel_IterateXPos.setArg(6, disp_range);
		Kernel_IterateXPos.setArg(7, SMOOTH_PENALTY);
		Kernel_IterateXPos.setArg(8, EDGE_PENALTY);
		err = queue.enqueueNDRangeKernel(Kernel_IterateXPos, cl::NullRange, cl::NDRange(height, dispRange), cl::NullRange, NULL, &xPos);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR XPos %1").arg(err), QMessageBox::Ok);
		xPos.wait();
	}
	else if (diry > 0) 
	{
		Kernel_IterateYPos.setArg(0, diry);
		Kernel_IterateYPos.setArg(1, bufferLeft);
		Kernel_IterateYPos.setArg(2, *bufferHierCost[0]);
		Kernel_IterateYPos.setArg(3, bufferSGMCost);
		Kernel_IterateYPos.setArg(4, nx);
		Kernel_IterateYPos.setArg(5, ny);
		Kernel_IterateYPos.setArg(6, disp_range);
		Kernel_IterateYPos.setArg(7, SMOOTH_PENALTY);
		Kernel_IterateYPos.setArg(8, EDGE_PENALTY);
		err = queue.enqueueNDRangeKernel(Kernel_IterateYPos, cl::NullRange, cl::NDRange(width, dispRange), cl::NullRange, NULL, &yPos);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR YPos %1").arg(err), QMessageBox::Ok);
		yPos.wait();
	}
	else if (dirx < 0)
	{
		Kernel_IterateXNeg.setArg(0, dirx);
		Kernel_IterateXNeg.setArg(1, bufferLeft);
		Kernel_IterateXNeg.setArg(2, *bufferHierCost[0]);
		Kernel_IterateXNeg.setArg(3, bufferSGMCost);
		Kernel_IterateXNeg.setArg(4, nx);
		Kernel_IterateXNeg.setArg(5, ny);
		Kernel_IterateXNeg.setArg(6, disp_range);
		Kernel_IterateXNeg.setArg(7, SMOOTH_PENALTY);
		Kernel_IterateXNeg.setArg(8, EDGE_PENALTY);
		err = queue.enqueueNDRangeKernel(Kernel_IterateXNeg, cl::NullRange, cl::NDRange(height, dispRange), cl::NullRange, NULL, &xNeg);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR XNeg %1").arg(err), QMessageBox::Ok);
		xNeg.wait();
	}
	else if (diry < 0)
	{
		Kernel_IterateYNeg.setArg(0, diry);
		Kernel_IterateYNeg.setArg(1, bufferLeft);
		Kernel_IterateYNeg.setArg(2, *bufferHierCost[0]);
		Kernel_IterateYNeg.setArg(3, bufferSGMCost);
		Kernel_IterateYNeg.setArg(4, nx);
		Kernel_IterateYNeg.setArg(5, ny);
		Kernel_IterateYNeg.setArg(6, disp_range);
		Kernel_IterateYNeg.setArg(7, SMOOTH_PENALTY);
		Kernel_IterateYNeg.setArg(8, EDGE_PENALTY);
		err = queue.enqueueNDRangeKernel(Kernel_IterateYNeg, cl::NullRange, cl::NDRange(width, dispRange), cl::NullRange, NULL, &yNeg);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR YNeg %1").arg(err), QMessageBox::Ok);
		yNeg.wait();
	}
}

void Final_Project::SAD_GPU()
{
	float DISCONT_COST = 0.5f;
	float CONT_COST = 0.02f;
	float SMOOTH_PENALTY = 0.7f;
	float EDGE_PENALTY = 0.9f;
	QString cb = ui.combo_box_optimization->currentText();
	QString cb2 = ui.combo_box_color->currentText();
	if (!imageLoaded)
	{
		QMessageBox::information(0, QString("NOTICE"), QString("Load image first"), QMessageBox::Ok);
		return;
	}
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//Start timer
	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);

	//Gaussian GPU
	Kernel_Gaussian.setArg(0, bufferLeft);
	Kernel_Gaussian.setArg(1, bufferRight);
	Kernel_Gaussian.setArg(2, bufferGaussianLeft);
	Kernel_Gaussian.setArg(3, bufferGaussianRight);
	err = queue.enqueueNDRangeKernel(Kernel_Gaussian, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &gaussian);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	gaussian.wait();


	//Sobel GPU
	Kernel_Sobel.setArg(0, bufferGaussianLeft);
	Kernel_Sobel.setArg(1, bufferGaussianRight);
	Kernel_Sobel.setArg(2, bufferSobelLeft);
	Kernel_Sobel.setArg(3, bufferSobelRight);
	err = queue.enqueueNDRangeKernel(Kernel_Sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &sobel);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	sobel.wait();

	//SAD GPU
	Kernel_SAD.setArg(0, bufferSobelLeft);
	Kernel_SAD.setArg(1, bufferSobelRight);
	Kernel_SAD.setArg(2, *bufferHierCost[0]);
	Kernel_SAD.setArg(3, width);
	Kernel_SAD.setArg(4, height);
	Kernel_SAD.setArg(5, dispRange);
	Kernel_SAD.setArg(6, mSize);
	//Execute kernel
	err = queue.enqueueNDRangeKernel(Kernel_SAD, cl::NullRange, cl::NDRange(width, height, dispRange), cl::NullRange, NULL, &sad);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR SAD %1").arg(err), QMessageBox::Ok);
	sad.wait();

	if (cb.toStdString() == "BP")
	{
		//BP GPU
		int i;
		int scale = 2;
		//Construct hier cost bottom up
		for (i = 1; i < HIER_LEVELS; i++)
		{
			Kernel_InitHierCost.setArg(0, *bufferHierCost[i - 1]);
			Kernel_InitHierCost.setArg(1, *bufferHierCost[i]);
			Kernel_InitHierCost.setArg(2, width / scale);
			Kernel_InitHierCost.setArg(3, height / scale);
			Kernel_InitHierCost.setArg(4, dispRange);
			//Execute kernel
			err = queue.enqueueNDRangeKernel(Kernel_InitHierCost, cl::NullRange, cl::NDRange(width / scale, height / scale, dispRange), cl::NullRange, NULL, &initHierCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR CONTRUCTING BP %1").arg(err), QMessageBox::Ok);
			initHierCost.wait();
			scale *= 2;
		}

		//Hier BP top down
		int dist = 1;

		for (int j = 1; j < HIER_LEVELS; j++)
		{
			dist *= 2; //2^(HIER_LEVEL-1)
		}

		for (int j = (HIER_LEVELS - 1); j >= 0; j--)
		{

			for (int i = 0; i < HIER_ITER; i++)
			{
				if (i % 2 == 0)
				{
					Kernel_HierBP.setArg(0, temp_m_u);
					Kernel_HierBP.setArg(1, temp_m_d);
					Kernel_HierBP.setArg(2, temp_m_l);
					Kernel_HierBP.setArg(3, temp_m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, m_u);
					Kernel_HierBP.setArg(6, m_d);
					Kernel_HierBP.setArg(7, m_l);
					Kernel_HierBP.setArg(8, m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
				else
				{
					Kernel_HierBP.setArg(0, m_u);
					Kernel_HierBP.setArg(1, m_d);
					Kernel_HierBP.setArg(2, m_l);
					Kernel_HierBP.setArg(3, m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, temp_m_u);
					Kernel_HierBP.setArg(6, temp_m_d);
					Kernel_HierBP.setArg(7, temp_m_l);
					Kernel_HierBP.setArg(8, temp_m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
			}

			if (j != 0)
			{
				//Use original width and height in the kernel pass parameter, but global size is based on level width and height
				Kernel_UpdateCostLayer.setArg(0, m_u);
				Kernel_UpdateCostLayer.setArg(1, m_d);
				Kernel_UpdateCostLayer.setArg(2, m_l);
				Kernel_UpdateCostLayer.setArg(3, m_r);
				Kernel_UpdateCostLayer.setArg(4, width);
				Kernel_UpdateCostLayer.setArg(5, height);
				Kernel_UpdateCostLayer.setArg(6, dispRange);
				Kernel_UpdateCostLayer.setArg(7, dist);
				err = queue.enqueueNDRangeKernel(Kernel_UpdateCostLayer, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &updateCostLayer);
				if (err != CL_SUCCESS)
					QMessageBox::information(0, QString("ERROR"), QString("ERROR UPDATE LAYER BP %1").arg(err), QMessageBox::Ok);
				updateCostLayer.wait();
			}
			dist /= 2;
		}

		//Final BP
		Kernel_FinalBP.setArg(0, m_u);
		Kernel_FinalBP.setArg(1, m_d);
		Kernel_FinalBP.setArg(2, m_l);
		Kernel_FinalBP.setArg(3, m_r);
		Kernel_FinalBP.setArg(4, *bufferHierCost[0]);
		Kernel_FinalBP.setArg(5, bufferZero);
		Kernel_FinalBP.setArg(6, width);
		Kernel_FinalBP.setArg(7, height);
		Kernel_FinalBP.setArg(8, dispRange);
		err = queue.enqueueNDRangeKernel(Kernel_FinalBP, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &finalBP);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR FINAL BP %1").arg(err), QMessageBox::Ok);
		finalBP.wait();
	}
	else if (cb.toStdString() == "SGM")
	{
		int dirx = 0, diry = 0; // The optimization direction in x and y respectively

		for (dirx = -1; dirx<2; dirx++) 
		{
			if (dirx == 0 && diry == 0) continue;

			iterateDirection(dirx, diry, width, height, dispRange, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width,height,dispRange), cl::NullRange, NULL, &sumCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost.wait();
		}

		dirx = 0;
		for (diry = -1; diry < 2; diry++) 
		{
			if (dirx == 0 && diry == 0) continue;

			iterateDirection(dirx, diry, width, height, dispRange, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width, height, dispRange), cl::NullRange, NULL, &sumCost2);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost2.wait();
		}
		*bufferHierCost[0] = bufferSGMAccumulate;
	}

	//WTA GPU
	Kernel_WTA.setArg(0, *bufferHierCost[0]);
	Kernel_WTA.setArg(1, bufferDepth);
	Kernel_WTA.setArg(2, width);
	Kernel_WTA.setArg(3, height);
	Kernel_WTA.setArg(4, dispRange);
	Kernel_WTA.setArg(5, 256 / dispRange);
	//Execute kernel
	err = queue.enqueueNDRangeKernel(Kernel_WTA, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &wta);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR WTA %1").arg(err), QMessageBox::Ok);
	wta.wait();

	//Copy device to host
	err = queue.enqueueReadBuffer(bufferDepth, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	depth_image.data = deviceToHost;
	
	//Stop timer
	QueryPerformanceCounter(&stop);
	timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;

	QMessageBox::information(0, QString("NOTICE"), QString("Depthmap from SAD GPU"), QMessageBox::Ok);
	QApplication::restoreOverrideCursor();
	Mat depth_color;
	switch (cb2.toInt())
	{
	case 1:
		applyColorMap(depth_image, depth_color, COLORMAP_BONE);
		break;
	case 2:
		applyColorMap(depth_image, depth_color, COLORMAP_JET);
		break;
	case 3:
		applyColorMap(depth_image, depth_color, COLORMAP_HOT);
		break;
	case 4:
		applyColorMap(depth_image, depth_color, COLORMAP_WINTER);
		break;
	}
	imwrite("SAD_BP.png", depth_color);
	QStereoDepth = MatToQTImage(depth_color);
	ui.stereo_depth_GPU->setPixmap(QPixmap::fromImage(QStereoDepth));
	ui.GPU_time->setText(QString::number(timespent) + " seconds");
	initOpenCLBuffer();
}

void Final_Project::ASW_GPU()
{
	float DISCONT_COST = 0.005f;
	float CONT_COST = 0.1f;
	float SMOOTH_PENALTY = 0.005f;
	float EDGE_PENALTY = 1.0f;
	QString cb = ui.combo_box_optimization->currentText();
	QString cb2 = ui.combo_box_color->currentText();
	if (!imageLoaded)
	{
		QMessageBox::information(0, QString("NOTICE"), QString("Load image first"), QMessageBox::Ok);
		return;
	}
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//Start timer
	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);

	//stereoWidth = width * 2;
	//bufferStereo = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, stereoWidth * height * sizeof(cl_uchar), frames.data, NULL);
	//bufferLeft = cl::Buffer(context, CL_MEM_READ_WRITE , totalPixels * sizeof(cl_uchar));
	//bufferRight = cl::Buffer(context, CL_MEM_READ_WRITE , totalPixels * sizeof(cl_uchar));

	//Kernel_Rectification_VGA.setArg(0, bufferStereo);
	//Kernel_Rectification_VGA.setArg(1, bufferLeft);
	//Kernel_Rectification_VGA.setArg(2, bufferRight);
	//Kernel_Rectification_VGA.setArg(3, stereoWidth);
	//Kernel_Rectification_VGA.setArg(4, height);
	//err = queue.enqueueNDRangeKernel(Kernel_Rectification_VGA, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &rectification);
	//if (err != CL_SUCCESS)
	//	QMessageBox::information(0, QString("ERROR"), QString("ERROR Rectification %1").arg(err), QMessageBox::Ok);
	//rectification.wait();

	//Mat save = Mat(height, width, CV_8UC1, cvScalar(0));
	//err = queue.enqueueReadBuffer(bufferLeft, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	//save.data = deviceToHost;
	//imwrite("RectificationLeft.png", save);

	//err = queue.enqueueReadBuffer(bufferRight, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	//save.data = deviceToHost;
	//imwrite("RectificationRight.png", save);

	//Gaussian GPU
	Kernel_Gaussian.setArg(0, bufferLeft);
	Kernel_Gaussian.setArg(1, bufferRight);
	Kernel_Gaussian.setArg(2, bufferGaussianLeft);
	Kernel_Gaussian.setArg(3, bufferGaussianRight);
	err = queue.enqueueNDRangeKernel(Kernel_Gaussian, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &gaussian);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	gaussian.wait();

	
	//Copy device to host
	/*Mat save = Mat(height, width, CV_8UC1, cvScalar(0));
	err = queue.enqueueReadBuffer(bufferGaussianLeft, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	save.data = deviceToHost;
 	imwrite("GaussianLeft.png", save);

	err = queue.enqueueReadBuffer(bufferGaussianRight, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	save.data = deviceToHost;
	imwrite("GaussianRight.png", save);*/

	//Sobel GPU
	Kernel_Sobel.setArg(0, bufferGaussianLeft);
	Kernel_Sobel.setArg(1, bufferGaussianRight);
	Kernel_Sobel.setArg(2, bufferSobelLeft);
	Kernel_Sobel.setArg(3, bufferSobelRight);
	err = queue.enqueueNDRangeKernel(Kernel_Sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &sobel);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	sobel.wait();

	//Copy device to host
	//err = queue.enqueueReadBuffer(bufferSobelLeft, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	//save.data = deviceToHost;
	//imwrite("SobelLeft.png", save);

	//err = queue.enqueueReadBuffer(bufferSobelRight, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	//save.data = deviceToHost;
	//imwrite("SobelRight.png", save);

	//ASW GPU
	Kernel_ASW.setArg(0, bufferSobelLeft);
	Kernel_ASW.setArg(1, bufferSobelRight);
	Kernel_ASW.setArg(2, *bufferHierCost[0]);
	Kernel_ASW.setArg(3, width);
	Kernel_ASW.setArg(4, height);
	Kernel_ASW.setArg(5, dispRange);
	Kernel_ASW.setArg(6, mSize_ASW);
	//Execute kernel
	err = queue.enqueueNDRangeKernel(Kernel_ASW, cl::NullRange, cl::NDRange(width, height, dispRange), cl::NullRange, NULL, &asw);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR ASW %1").arg(err), QMessageBox::Ok);
	asw.wait();

	if (cb.toStdString() == "BP")
	{
		//BP GPU
		int i;
		int scale = 2;
		//Construct hier cost bottom up
		for (i = 1; i < HIER_LEVELS; i++)
		{
			Kernel_InitHierCost.setArg(0, *bufferHierCost[i - 1]);
			Kernel_InitHierCost.setArg(1, *bufferHierCost[i]);
			Kernel_InitHierCost.setArg(2, width / scale);
			Kernel_InitHierCost.setArg(3, height / scale);
			Kernel_InitHierCost.setArg(4, dispRange);
			//Execute kernel
			err = queue.enqueueNDRangeKernel(Kernel_InitHierCost, cl::NullRange, cl::NDRange(width / scale, height / scale, dispRange), cl::NullRange, NULL, &initHierCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR CONTRUCTING BP %1").arg(err), QMessageBox::Ok);
			initHierCost.wait();
			scale *= 2;
		}

		//Hier BP top down
		int dist = 1;

		for (int j = 1; j < HIER_LEVELS; j++)
		{
			dist *= 2; //2^(HIER_LEVEL-1)
		}

		for (int j = (HIER_LEVELS - 1); j >= 0; j--)
		{

			for (int i = 0; i < HIER_ITER; i++)
			{
				if (i % 2 == 0)
				{
					Kernel_HierBP.setArg(0, temp_m_u);
					Kernel_HierBP.setArg(1, temp_m_d);
					Kernel_HierBP.setArg(2, temp_m_l);
					Kernel_HierBP.setArg(3, temp_m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, m_u);
					Kernel_HierBP.setArg(6, m_d);
					Kernel_HierBP.setArg(7, m_l);
					Kernel_HierBP.setArg(8, m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
				else
				{
					Kernel_HierBP.setArg(0, m_u);
					Kernel_HierBP.setArg(1, m_d);
					Kernel_HierBP.setArg(2, m_l);
					Kernel_HierBP.setArg(3, m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, temp_m_u);
					Kernel_HierBP.setArg(6, temp_m_d);
					Kernel_HierBP.setArg(7, temp_m_l);
					Kernel_HierBP.setArg(8, temp_m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
			}

			if (j != 0)
			{
				//Use original width and height in the kernel pass parameter, but global size is based on level width and height
				Kernel_UpdateCostLayer.setArg(0, m_u);
				Kernel_UpdateCostLayer.setArg(1, m_d);
				Kernel_UpdateCostLayer.setArg(2, m_l);
				Kernel_UpdateCostLayer.setArg(3, m_r);
				Kernel_UpdateCostLayer.setArg(4, width);
				Kernel_UpdateCostLayer.setArg(5, height);
				Kernel_UpdateCostLayer.setArg(6, dispRange);
				Kernel_UpdateCostLayer.setArg(7, dist);
				err = queue.enqueueNDRangeKernel(Kernel_UpdateCostLayer, cl::NullRange, cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &updateCostLayer);
				if (err != CL_SUCCESS)
					QMessageBox::information(0, QString("ERROR"), QString("ERROR UPDATE LAYER BP %1").arg(err), QMessageBox::Ok);
				updateCostLayer.wait();
			}
			dist /= 2;
		}

		//Final BP
		Kernel_FinalBP.setArg(0, m_u);
		Kernel_FinalBP.setArg(1, m_d);
		Kernel_FinalBP.setArg(2, m_l);
		Kernel_FinalBP.setArg(3, m_r);
		Kernel_FinalBP.setArg(4, *bufferHierCost[0]);
		Kernel_FinalBP.setArg(5, bufferZero);
		Kernel_FinalBP.setArg(6, width);
		Kernel_FinalBP.setArg(7, height);
		Kernel_FinalBP.setArg(8, dispRange);
		err = queue.enqueueNDRangeKernel(Kernel_FinalBP, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &finalBP);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR FINAL BP %1").arg(err), QMessageBox::Ok);
		finalBP.wait();
	}
	else if (cb.toStdString() == "SGM")
	{
		int dirx = 0, diry = 0; // The optimization direction in x and y respectively

		for (dirx = -1; dirx<2; dirx++)
		{
			if (dirx == 0 && diry == 0) continue;
			iterateDirection(dirx, diry, width, height, dispRange, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width, height, dispRange), cl::NullRange, NULL, &sumCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost.wait();
		}

		dirx = 0;
		for (diry = -1; diry < 2; diry++)
		{
			if (dirx == 0 && diry == 0) continue;
			iterateDirection(dirx, diry, width, height, dispRange, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width, height, dispRange), cl::NullRange, NULL, &sumCost2);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost2.wait();
		}
		*bufferHierCost[0] = bufferSGMAccumulate;
	}

	//WTA GPU
	Kernel_WTA.setArg(0, *bufferHierCost[0]);
	Kernel_WTA.setArg(1, bufferDepth);
	Kernel_WTA.setArg(2, width);
	Kernel_WTA.setArg(3, height);
	Kernel_WTA.setArg(4, dispRange);
	Kernel_WTA.setArg(5, 256/dispRange);
	//Execute kernel
	err = queue.enqueueNDRangeKernel(Kernel_WTA, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &wta);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR WTA %1").arg(err), QMessageBox::Ok);
	wta.wait();

	//Copy device to host
	err = queue.enqueueReadBuffer(bufferDepth, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	depth_image.data = deviceToHost;
	imwrite("depth.png", depth_image);

	//Copy device to host
	//float *deviceToHost2 = new float[totalPixels*dispRange]();
	//err = queue.enqueueReadBuffer(*bufferHierCost[0], CL_TRUE, 0, (totalPixels) * sizeof(float), deviceToHost2);

	//for (int k = 0; k < dispRange; k++)
	//{
	//	ofstream file;
	//	file.open("cost_at_disp_" + std::to_string(k + 1) + ".txt");
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{

	//			file << (deviceToHost2[dispRange * (width * i + j) + k]);
	//			file << "\t";
	//		}
	//		file << "\n";
	//	}
	//	file.close();
	//}
	

	//Stop timer
	QueryPerformanceCounter(&stop);
	timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;

	QMessageBox::information(0, QString("NOTICE"), QString("Depthmap from ASW GPU"), QMessageBox::Ok);
	QApplication::restoreOverrideCursor();
	Mat depth_color;
	switch (cb2.toInt())
	{
	case 1:
		applyColorMap(depth_image, depth_color, COLORMAP_BONE);
		break;
	case 2:
		applyColorMap(depth_image, depth_color, COLORMAP_JET);
		break;
	case 3:
		applyColorMap(depth_image, depth_color, COLORMAP_HOT);
		break;
	case 4:
		applyColorMap(depth_image, depth_color, COLORMAP_WINTER);
		break;
	}
	imwrite("ASW_BP.png", depth_color);
	QStereoDepth = MatToQTImage(depth_color);
	ui.stereo_depth_GPU->setPixmap(QPixmap::fromImage(QStereoDepth));
	ui.GPU_time->setText(QString::number(timespent) + " seconds");
	initOpenCLBuffer();
}

void Final_Project::displayVideo()
{
	QString cb = ui.combo_box->currentText();
	QString cb2 = ui.combo_box_cost->currentText();
	QString cb3 = ui.combo_box_resolution->currentText();
	QString cb4 = ui.combo_box_optimization->currentText();
	QString cb5 = ui.combo_box_color->currentText();
	//Start timer
	QueryPerformanceFrequency(&frequencies);
	QueryPerformanceCounter(&start);

	cap >> frame;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	if (firstFrame == true)
	{
		//Initialization OpenCL
		int error = initOpenCL();
		if ( error != 1) {
			QMessageBox::information(0, QString("ERROR"), QString("Failed to initialize OpenCL %1").arg(error), QMessageBox::Ok);
		}
		else
		{
			QMessageBox::information(0, QString("NOTICE"), QString("Successfully to initialize OpenCL"), QMessageBox::Ok);
		}

		stereoWidth = frame.cols;
		width = stereoWidth / 2;
		height = frame.rows;
		totalPixels = width * height;
		initOpenCLBufferVideo();
		firstFrame = false;
		QMessageBox::information(0, QString("NOTICE"), QString("WIDTH: %1  HEIGHT: %2").arg(width).arg(height), QMessageBox::Ok);
	}

	//Copy frame from stereo camera
	bufferStereo = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, stereoWidth * height * sizeof(cl_uchar), frame_gray.data, NULL);

	//Rectification GPU
	if (cb3.toStdString() == "HD")
	{
		Kernel_Rectification.setArg(0, bufferStereo);
		Kernel_Rectification.setArg(1, bufferLeft);
		Kernel_Rectification.setArg(2, bufferRight);
		Kernel_Rectification.setArg(3, stereoWidth);
		Kernel_Rectification.setArg(4, height);
		err = queue.enqueueNDRangeKernel(Kernel_Rectification, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &rectification);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR Rectification %1").arg(err), QMessageBox::Ok);
		rectification.wait();
	}
	else
	{
		Kernel_Rectification_VGA.setArg(0, bufferStereo);
		Kernel_Rectification_VGA.setArg(1, bufferLeft);
		Kernel_Rectification_VGA.setArg(2, bufferRight);
		Kernel_Rectification_VGA.setArg(3, stereoWidth);
		Kernel_Rectification_VGA.setArg(4, height);
		err = queue.enqueueNDRangeKernel(Kernel_Rectification_VGA, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &rectification);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR Rectification %1").arg(err), QMessageBox::Ok);
		rectification.wait();
	}


	if (cb.toStdString() == "Rectified Frame")
	{
		//Get the rectified frames
		err = queue.enqueueReadBuffer(bufferLeft, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
		left_rectified.data = deviceToHost;
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR READ LEFT %1").arg(err), QMessageBox::Ok);
		err = queue.enqueueReadBuffer(bufferRight, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
		right_rectified.data = deviceToHost;
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR READ RIGHT %1").arg(err), QMessageBox::Ok);
		qStereoLeft = MatToQTImage(left_rectified);
		ui.stereo_left->setPixmap(QPixmap::fromImage(qStereoLeft));
		qStereoRight = MatToQTImage(right_rectified);
		ui.stereo_right->setPixmap(QPixmap::fromImage(qStereoRight));

	}

	else if (cb.toStdString() == "Stereo Frame")
	{
		/*left_image = frame_gray(Rect(0, 0, frame.cols / 2, frame.rows));
		right_image = frame_gray(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));*/
		left_image = frame(Rect(0, 0, frame.cols / 2, frame.rows));
		right_image = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
		qStereoLeft = MatToQTImage(left_image);
		ui.stereo_left->setPixmap(QPixmap::fromImage(qStereoLeft));
		qStereoRight = MatToQTImage(right_image);
		ui.stereo_right->setPixmap(QPixmap::fromImage(qStereoRight));
	}
	else
	{
		ui.stereo_left->clear();
		ui.stereo_right->clear();
	}

	//Gaussian GPU
	Kernel_Gaussian.setArg(0, bufferLeft);
	Kernel_Gaussian.setArg(1, bufferRight);
	Kernel_Gaussian.setArg(2, bufferGaussianLeft);
	Kernel_Gaussian.setArg(3, bufferGaussianRight);
	err = queue.enqueueNDRangeKernel(Kernel_Gaussian, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &gaussian);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	gaussian.wait();

	//Sobel GPU
	Kernel_Sobel.setArg(0, bufferGaussianLeft);
	Kernel_Sobel.setArg(1, bufferGaussianRight);
	Kernel_Sobel.setArg(2, bufferSobelLeft);
	Kernel_Sobel.setArg(3, bufferSobelRight);
	err = queue.enqueueNDRangeKernel(Kernel_Sobel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &sobel);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Gaussian %1").arg(err), QMessageBox::Ok);
	sobel.wait();
	float DISCONT_COST;
	float CONT_COST;
	float SMOOTH_PENALTY;
	float EDGE_PENALTY;
	if (cb2.toStdString() == "SAD")
	{
		SMOOTH_PENALTY = 0.2f;
		EDGE_PENALTY = 2.0f;
		//SAD GPU
		Kernel_SAD.setArg(0, bufferGaussianLeft);
		Kernel_SAD.setArg(1, bufferGaussianRight);
		Kernel_SAD.setArg(2, *bufferHierCost[0]);
		Kernel_SAD.setArg(3, width);
		Kernel_SAD.setArg(4, height);
		Kernel_SAD.setArg(5, dispRange_video);
		Kernel_SAD.setArg(6, mSize);
		//Execute kernel
		err = queue.enqueueNDRangeKernel(Kernel_SAD, cl::NullRange, cl::NDRange(width, height, dispRange_video), cl::NullRange, NULL, &sad);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR SAD %1").arg(err), QMessageBox::Ok);
		sad.wait();
		DISCONT_COST = 0.6f;
		CONT_COST = 1.5f;
	}
	else if (cb2.toStdString() == "ASW")
	{
		SMOOTH_PENALTY = 0.001f;
		EDGE_PENALTY = 0.06f;
		//ASW GPU
		Kernel_ASW.setArg(0, bufferGaussianLeft);
		Kernel_ASW.setArg(1, bufferGaussianRight);
		Kernel_ASW.setArg(2, *bufferHierCost[0]);
		Kernel_ASW.setArg(3, width);
		Kernel_ASW.setArg(4, height);
		Kernel_ASW.setArg(5, dispRange_video);
		Kernel_ASW.setArg(6, mSize * 2);
		//Execute kernel
		err = queue.enqueueNDRangeKernel(Kernel_ASW, cl::NullRange, cl::NDRange(width, height, dispRange_video), cl::NullRange, NULL, &asw);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR ASW %1").arg(err), QMessageBox::Ok);
		asw.wait();
		DISCONT_COST = 0.03f;
		CONT_COST = 0.2f;
	}

	if (cb4.toStdString() == "BP")
	{
		//BP GPU
		int i;
		int scale = 2;
		//Construct hier cost bottom up
		for (i = 1; i < HIER_LEVELS; i++)
		{
			Kernel_InitHierCost.setArg(0, *bufferHierCost[i - 1]);
			Kernel_InitHierCost.setArg(1, *bufferHierCost[i]);
			Kernel_InitHierCost.setArg(2, width / scale);
			Kernel_InitHierCost.setArg(3, height / scale);
			Kernel_InitHierCost.setArg(4, dispRange_video);
			//Execute kernel
			err = queue.enqueueNDRangeKernel(Kernel_InitHierCost, cl::NullRange, cl::NDRange(width / scale, height / scale, dispRange_video), cl::NullRange, NULL, &initHierCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR CONTRUCTING BP %1").arg(err), QMessageBox::Ok);
			initHierCost.wait();
			scale *= 2;
		}

		//Hier BP top down
		int dist = 1;

		for (int j = 1; j < HIER_LEVELS; j++)
		{
			dist *= 2; //2^(HIER_LEVEL-1)
		}

		for (int j = (HIER_LEVELS - 1); j >= 0; j--)
		{

			for (int i = 0; i < HIER_ITER/2; i++)
			{
				if (i % 2 == 0)
				{
					Kernel_HierBP.setArg(0, temp_m_u);
					Kernel_HierBP.setArg(1, temp_m_d);
					Kernel_HierBP.setArg(2, temp_m_l);
					Kernel_HierBP.setArg(3, temp_m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, m_u);
					Kernel_HierBP.setArg(6, m_d);
					Kernel_HierBP.setArg(7, m_l);
					Kernel_HierBP.setArg(8, m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange_video);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NDRange(150, 150), cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
				else
				{
					Kernel_HierBP.setArg(0, m_u);
					Kernel_HierBP.setArg(1, m_d);
					Kernel_HierBP.setArg(2, m_l);
					Kernel_HierBP.setArg(3, m_r);
					Kernel_HierBP.setArg(4, *bufferHierCost[j]);
					Kernel_HierBP.setArg(5, temp_m_u);
					Kernel_HierBP.setArg(6, temp_m_d);
					Kernel_HierBP.setArg(7, temp_m_l);
					Kernel_HierBP.setArg(8, temp_m_r);
					Kernel_HierBP.setArg(9, bufferZero);
					Kernel_HierBP.setArg(10, width / dist);
					Kernel_HierBP.setArg(11, height / dist);
					Kernel_HierBP.setArg(12, dispRange_video);
					Kernel_HierBP.setArg(13, dist);
					Kernel_HierBP.setArg(14, DISCONT_COST);
					Kernel_HierBP.setArg(15, CONT_COST);
					err = queue.enqueueNDRangeKernel(Kernel_HierBP, cl::NDRange(150, 150), cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &hierBP);
					if (err != CL_SUCCESS)
						QMessageBox::information(0, QString("ERROR"), QString("ERROR MSG BP %1").arg(err), QMessageBox::Ok);
					hierBP.wait();
				}
			}

			if (j != 0)
			{
				//Use original width and height in the kernel pass parameter, but global size is based on level width and height
				Kernel_UpdateCostLayer.setArg(0, m_u);
				Kernel_UpdateCostLayer.setArg(1, m_d);
				Kernel_UpdateCostLayer.setArg(2, m_l);
				Kernel_UpdateCostLayer.setArg(3, m_r);
				Kernel_UpdateCostLayer.setArg(4, width);
				Kernel_UpdateCostLayer.setArg(5, height);
				Kernel_UpdateCostLayer.setArg(6, dispRange_video);
				Kernel_UpdateCostLayer.setArg(7, dist);
				err = queue.enqueueNDRangeKernel(Kernel_UpdateCostLayer, cl::NDRange(150, 150), cl::NDRange(width / dist, height / dist), cl::NullRange, NULL, &updateCostLayer);
				if (err != CL_SUCCESS)
					QMessageBox::information(0, QString("ERROR"), QString("ERROR UPDATE LAYER BP %1").arg(err), QMessageBox::Ok);
				updateCostLayer.wait();
			}
			dist /= 2;
		}

		//Final BP
		Kernel_FinalBP.setArg(0, m_u);
		Kernel_FinalBP.setArg(1, m_d);
		Kernel_FinalBP.setArg(2, m_l);
		Kernel_FinalBP.setArg(3, m_r);
		Kernel_FinalBP.setArg(4, *bufferHierCost[0]);
		Kernel_FinalBP.setArg(5, bufferZero);
		Kernel_FinalBP.setArg(6, width);
		Kernel_FinalBP.setArg(7, height);
		Kernel_FinalBP.setArg(8, dispRange_video);
		err = queue.enqueueNDRangeKernel(Kernel_FinalBP, cl::NDRange(150, 150), cl::NDRange(width, height), cl::NullRange, NULL, &finalBP);
		if (err != CL_SUCCESS)
			QMessageBox::information(0, QString("ERROR"), QString("ERROR FINAL BP %1").arg(err), QMessageBox::Ok);
		finalBP.wait();
	}
	else if (cb4.toStdString() == "SGM")
	{
		int dirx = 0, diry = 0; // The optimization direction in x and y respectively

		for (dirx = -1; dirx<2; dirx++)
		{
			if (dirx == 0 && diry == 0) continue;
			bufferSGMCost = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * dispRange_video * sizeof(cl_float), zeros, NULL);
			iterateDirection(dirx, diry, width, height, dispRange_video, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange_video);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width, height, dispRange_video), cl::NullRange, NULL, &sumCost);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost.wait();
		}

		dirx = 0;
		for (diry = -1; diry < 2; diry++)
		{
			if (dirx == 0 && diry == 0) continue;
			bufferSGMCost = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * dispRange_video * sizeof(cl_float), zeros, NULL);
			iterateDirection(dirx, diry, width, height, dispRange_video, SMOOTH_PENALTY, EDGE_PENALTY);

			Kernel_SumCost.setArg(0, bufferSGMAccumulate);
			Kernel_SumCost.setArg(1, bufferSGMCost);
			Kernel_SumCost.setArg(2, width);
			Kernel_SumCost.setArg(3, dispRange_video);
			err = queue.enqueueNDRangeKernel(Kernel_SumCost, cl::NullRange, cl::NDRange(width, height, dispRange_video), cl::NullRange, NULL, &sumCost2);
			if (err != CL_SUCCESS)
				QMessageBox::information(0, QString("ERROR"), QString("ERROR SumCost %1").arg(err), QMessageBox::Ok);
			sumCost2.wait();
		}
		*bufferHierCost[0] = bufferSGMAccumulate;
	}

	//WTA GPU
	Kernel_WTA.setArg(0, *bufferHierCost[0]);
	Kernel_WTA.setArg(1, bufferDepth);
	Kernel_WTA.setArg(2, width);
	Kernel_WTA.setArg(3, height);
	Kernel_WTA.setArg(4, dispRange_video);
	Kernel_WTA.setArg(5, 256 / dispRange_video);
	//Execute kernel
	err = queue.enqueueNDRangeKernel(Kernel_WTA, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &wta);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR WTA %1").arg(err), QMessageBox::Ok);
	wta.wait();

	//Warp back depth
	Kernel_Rectification_back.setArg(0, bufferDepth);
	Kernel_Rectification_back.setArg(1, bufferDepthWarp);
	Kernel_Rectification_back.setArg(2, width);
	Kernel_Rectification_back.setArg(3, height);
	err = queue.enqueueNDRangeKernel(Kernel_Rectification_back, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &rectification_back);
	if (err != CL_SUCCESS)
		QMessageBox::information(0, QString("ERROR"), QString("ERROR Rectification Back %1").arg(err), QMessageBox::Ok);
	rectification_back.wait();

	//Copy device to host
	err = queue.enqueueReadBuffer(bufferDepthWarp, CL_TRUE, 0, (totalPixels) * sizeof(unsigned char), deviceToHost);
	depth_image.data = deviceToHost;

	//Stop timer
	QueryPerformanceCounter(&stop);
	timespent = (double)(stop.QuadPart - start.QuadPart) / (double)frequencies.QuadPart;
	timespent = 1 / timespent;
	Mat depth_color;
	switch (cb5.toInt())
	{
	case 1:
		applyColorMap(depth_image, depth_color, COLORMAP_BONE);
		break;
	case 2:
		applyColorMap(depth_image, depth_color, COLORMAP_JET);
		break;
	case 3:
		applyColorMap(depth_image, depth_color, COLORMAP_HOT);
		break;
	case 4:
		applyColorMap(depth_image, depth_color, COLORMAP_WINTER);
		break;
	}
	QStereoDepth = MatToQTImage(depth_color);
	ui.stereo_depth_GPU->setPixmap(QPixmap::fromImage(QStereoDepth));
	ui.GPU_time->setText(QString::number(timespent) + " FPS");
}

void Final_Project::captureFrame()
{
	if (!cap.isOpened())
	{
		QMessageBox::information(0, QString("ERROR"), QString("No camera detected"), QMessageBox::Ok);
		return;
	}

	QString cb = ui.combo_box_resolution->currentText();
	if (cb.toStdString() == "HD")
	{
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 2560);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 960);
	}
	else if (cb.toStdString() == "VGA")
	{
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640);
	}
	
	ui.stereo_left->clear();
	ui.stereo_right->clear();
	ui.stereo_depth_CPU->clear();
	qTimer = new QTimer(this);
	connect(qTimer, SIGNAL(timeout()), this, SLOT(displayVideo()));
	qTimer->start(25);
}
