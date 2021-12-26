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
	ifstream fin("calibdata.txt");             /* �궨����ͼ���ļ���·�� */
	string old_image_path = "Pic2_4_gaitubao_657x493";          /*ԭͼ����ļ�·��*/
	string new_image_path="new_image.bmp";            /*���κ����ͼ��·��*/
	ofstream fout("caliberation_result.txt");  /* ����궨������ļ� */

	// ��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
	int image_count = 0;  /* ͼ������ */
	Size image_size;      /* ͼ��ĳߴ� */
	Size board_size = Size(7,7);             /* �궨����ÿ�С��еĽǵ��� */
	vector<Point2f> image_points_buf;         /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	vector<vector<Point2f>> image_points_seq; /* �����⵽�����нǵ� */
	string filename;      // ͼƬ��
	vector<string> filenames;

	while (getline(fin, filename))
	{
		++image_count;
		Mat imageInput = imread(filename);
		
		filenames.push_back(filename);
		//cout << "the part one is over��" << endl;
		// �����һ��ͼƬʱ��ȡͼƬ��С
		if (image_count == 1)
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "the width is "<<image_size.width<<"the image_size height is "<<image_size.height<<endl;
		}

		//int num = findCirclesGrid(imageInput, board_size, image_points_buf, cv::CALIB_CB_SYMMETRIC_GRID);
		//cout << "the num is " << num << endl;

		/* ��ȡ�ǵ� */
		if (0 == findCirclesGrid(imageInput, board_size, image_points_buf, cv::CALIB_CB_SYMMETRIC_GRID))
			//  0 == findCirclesGrid(imageInput, board_size, image_points, cv::CALIB_CB_ASYMMETRIC_GRID, blobDetector);
		{
			cout << "can not find chessboard corners!\n";  // �Ҳ����ǵ�
			exit(1);
		}
		else
		{
			cout << "the part two is over��" << endl;
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // ת�Ҷ�ͼ

			/* �����ؾ�ȷ�� */
			// image_points_buf ��ʼ�Ľǵ�����������ͬʱ��Ϊ����������λ�õ����
			// Size(5,5) �������ڴ�С
			// ��-1��-1����ʾû������
			// TermCriteria �ǵ�ĵ������̵���ֹ����, ����Ϊ���������ͽǵ㾫�����ߵ����
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			image_points_seq.push_back(image_points_buf);  // ���������ؽǵ�

			/* ��ͼ������ʾ�ǵ�λ�� */
			drawChessboardCorners(view_gray, board_size, image_points_buf, false); // ������ͼƬ�б�ǽǵ�
			
			//imshow("Camera Calibration", view_gray);       // ��ʾͼƬ
			//string path = "�궨���" + to_string(image_count);
			//auto path_2 = path.c_str();
			//imwrite("test", view_gray);
			waitKey(50); //��ͣ0.5S
		}
	}
	int CornerNum = board_size.width * board_size.height;  // ÿ��ͼƬ���ܵĽǵ���

	

	//-------------������������궨------------------

	/*������ά��Ϣ*/
	float square_size = 17.2525;         /* ʵ�ʲ����õ��ı궨����ͬһ����������Բ�İ�Բ�ľ� */
	vector<vector<Point3f>> object_points;   /* ����궨���Ͻǵ����ά���� */

	/*�������*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ������ڲ������� */
	vector<int> point_counts;   // ÿ��ͼ���нǵ������
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* �������5������ϵ����k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;      /* ÿ��ͼ�����ת���� */
	vector<Mat> rvecsMat;      /* ÿ��ͼ���ƽ������ */

	/* ��ʼ���궨���Ͻǵ����ά���� */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;

				/* ����궨�������������ϵ��z=0��ƽ���� */
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

	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}

	/* ��ʼ�궨 */
	// object_points ��������ϵ�еĽǵ����ά����
	// image_points_seq ÿһ���ڽǵ��Ӧ��ͼ�������
	// image_size ͼ������سߴ��С
	// cameraMatrix ������ڲξ���
	// distCoeffs ���������ϵ��
	// rvecsMat �������ת����
	// tvecsMat �����λ������
	// 0 �궨ʱ�����õ��㷨
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3);

	//------------------------�궨���------------------------------------

	// -------------------�Ա궨�����������------------------------------

	double total_err = 0.0;         /* ����ͼ���ƽ�������ܺ� */
	double err = 0.0;               /* ÿ��ͼ���ƽ����� */
	vector<Point2f> image_points2;  /* �������¼���õ���ͶӰ�� */
	fout << "ÿ��ͼ��ı궨��\n";

	for (i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];

		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << image_points2Mat << endl;
	}
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;

	//-------------------------�������---------------------------------------------

	//-----------------------���涨����-------------------------------------------
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ����ÿ��ͼ�����ת���� */
	fout << "����ڲ�������" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "����ϵ����\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << tvecsMat[i] << endl;

		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		Rodrigues(tvecsMat[i], rotation_matrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotation_matrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	fout << endl;

	//--------------------�궨����������-------------------------------

	//----------------------��ʾ������--------------------------------

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

	//��ԭͼ����б���
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