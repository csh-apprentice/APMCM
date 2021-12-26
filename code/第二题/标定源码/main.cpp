#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;


int main()
{
	ifstream fin("calibdata.txt");             /* 标定所用图像文件的路径 */
	string old_image_path = "Pic2_4_gaitubao_657x493";          /*原图像的文件路径*/
	string new_image_path="new_image.bmp";            /*变形后的新图像路径*/
	ofstream fout("caliberation_result.txt");  /* 保存标定结果的文件 */

	// 读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
	int image_count = 0;  /* 图像数量 */
	Size image_size;      /* 图像的尺寸 */
	Size board_size = Size(7,7);             /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;         /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
	string filename;      // 图片名
	vector<string> filenames;

	while (getline(fin, filename))
	{
		++image_count;
		Mat imageInput = imread(filename);
		
		filenames.push_back(filename);
		//cout << "the part one is over！" << endl;
		// 读入第一张图片时获取图片大小
		if (image_count == 1)
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "the width is "<<image_size.width<<"the image_size height is "<<image_size.height<<endl;
		}

		//int num = findCirclesGrid(imageInput, board_size, image_points_buf, cv::CALIB_CB_SYMMETRIC_GRID);
		//cout << "the num is " << num << endl;

		/* 提取角点 */
		if (0 == findCirclesGrid(imageInput, board_size, image_points_buf, cv::CALIB_CB_SYMMETRIC_GRID))
			//  0 == findCirclesGrid(imageInput, board_size, image_points, cv::CALIB_CB_ASYMMETRIC_GRID, blobDetector);
		{
			cout << "can not find chessboard corners!\n";  // 找不到角点
			exit(1);
		}
		else
		{
			cout << "the part two is over！" << endl;
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // 转灰度图

			/* 亚像素精确化 */
			// image_points_buf 初始的角点坐标向量，同时作为亚像素坐标位置的输出
			// Size(5,5) 搜索窗口大小
			// （-1，-1）表示没有死区
			// TermCriteria 角点的迭代过程的终止条件, 可以为迭代次数和角点精度两者的组合
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			image_points_seq.push_back(image_points_buf);  // 保存亚像素角点

			/* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray, board_size, image_points_buf, false); // 用于在图片中标记角点
			
			//imshow("Camera Calibration", view_gray);       // 显示图片
			//string path = "标定结果" + to_string(image_count);
			//auto path_2 = path.c_str();
			//imwrite("test", view_gray);
			waitKey(50); //暂停0.5S
		}
	}
	int CornerNum = board_size.width * board_size.height;  // 每张图片上总的角点数

	

	//-------------以下是摄像机标定------------------

	/*棋盘三维信息*/
	float square_size = 17.2525;         /* 实际测量得到的标定板上同一行相邻两个圆的半圆心距 */
	vector<vector<Point3f>> object_points;   /* 保存标定板上角点的三维坐标 */

	/*内外参数*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* 摄像机内参数矩阵 */
	vector<int> point_counts;   // 每幅图像中角点的数量
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;      /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat;      /* 每幅图像的平移向量 */

	/* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;

				/* 假设标定板放在世界坐标系中z=0的平面上 */
				if (i % 2 == 0) {
					realPoint.x = i * square_size;
					realPoint.y = (2 * j) * square_size;
					realPoint.z = 0;
				}
				else {
					realPoint.x = i * square_size;
					realPoint.y = (2 * j + 1) * square_size;
					realPoint.z = 0;
				}
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}

	/* 开始标定 */
	// object_points 世界坐标系中的角点的三维坐标
	// image_points_seq 每一个内角点对应的图像坐标点
	// image_size 图像的像素尺寸大小
	// cameraMatrix 输出，内参矩阵
	// distCoeffs 输出，畸变系数
	// rvecsMat 输出，旋转向量
	// tvecsMat 输出，位移向量
	// 0 标定时所采用的算法
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3);

	//------------------------标定完成------------------------------------

	// -------------------对标定结果进行评价------------------------------

	double total_err = 0.0;         /* 所有图像的平均误差的总和 */
	double err = 0.0;               /* 每幅图像的平均误差 */
	vector<Point2f> image_points2;  /* 保存重新计算得到的投影点 */
	fout << "每幅图像的标定误差：\n";

	for (i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];

		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << image_points2Mat << endl;
	}
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;

	//-------------------------评价完成---------------------------------------------

	//-----------------------保存定标结果-------------------------------------------
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* 保存每幅图像的旋转矩阵 */
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << tvecsMat[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(tvecsMat[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	fout << endl;

	//--------------------标定结果保存结束-------------------------------

	//----------------------显示定标结果--------------------------------

	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	string imageFileName;
	std::stringstream StrStm;
	for (int i = 0; i != image_count; i++)
	{
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		Mat imageSource = imread(filenames[i]);
		Mat newimage = imageSource.clone();
		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
		StrStm.clear();
		imageFileName.clear();
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite(imageFileName, newimage);
	}

	//对原图像进行变形
	initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
	Mat imageSource = imread(old_image_path);
	//image_size.width = imageSource.cols;
	//image_size.height = imageSource.rows;
	Mat newimage = imageSource.clone();
	cout << "fine here!" << endl;
	remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
	cout << "not fine here!" << endl;
	imwrite(new_image_path, newimage);

	fin.close();
	fout.close();
	//fout_image.close();
	return 0;
}