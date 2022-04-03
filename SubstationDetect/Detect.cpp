#include "Detect.h"
Mat MatAnd(Mat src, Mat mask) {
	Mat tmp = src.clone();
	for (int m = 0; m < src.rows; m++)
		for (int n = 0; n < src.cols; n++)
			tmp.at<float>(m, n) = src.at<float>(m, n) * mask.at<float>(m, n);
	return tmp;
}
Mat DisWhiteBorder(Mat in) {
	Mat ori = in.clone();
	cvtColor(in, in, COLOR_RGB2GRAY);
	threshold(in, in, 200, 255, THRESH_BINARY);
	Point lt(0, 0), rb(in.cols-1, in.rows-1);
	while (lt.x < in.cols / 2) {
		bool is_black = false;
		for (int y = 0; y < in.rows; y++) 
			if (in.at<uchar>(y, lt.x) == 0) {
				is_black = true;
			}
		if (is_black) break;
		lt.x++;
	}
	while (lt.y < in.cols / 2) {
		bool is_black = false;
		for (int x = 0; x < in.cols; x++)
			if (in.at<uchar>(lt.y, x) == 0) {
				is_black = true;
			}
		if (is_black) break;
		lt.y++;
	}
	while (rb.x > in.cols / 2) {
		bool is_black = false;
		for (int y = in.rows-1; y >0; y--)
			if (in.at<uchar>(y, rb.x) == 0) {
				is_black = true;
			}
		if (is_black) break;
		rb.x--;
	}
	while (rb.y > in.cols / 2) {
		bool is_black = false;
		for (int x = in.cols-1; x > 0; x--)
			if (in.at<uchar>(rb.y, x) == 0) {
				is_black = true;
			}
		if (is_black) break;
		rb.y--;
	}
	lt.x -= 5;
	lt.y -= 5;
	rb.x += 5;
	rb.y += 5;
	Rect ret(lt, rb);
	//imshow("2", in(ret));
	//waitKey(0);
	Mat out = ori(ret);
	return out;
}
void LableUp(String path) {
	vector<String> imgVessel;
	PathFetch(imgVessel, "jpg", path);
	for (int index = 0; index < imgVessel.size(); index++) {
		Mat show = imread(imgVessel[index]);
		String name = imgVessel[index].substr(0, imgVessel[index].length() - 4);
		//resizeWindow(imgVessel[index], Size(300, 300));
		int lable;
		cout << "ÕýÔÚ¶Ô£º" << imgVessel[index] << endl;
		cin >> lable;
		char buffer[512];
		strcpy_s(buffer, imgVessel[index].c_str());
		remove(buffer);
		name += "_" + to_string(lable) + "_.jpg";
		imwrite(name, show);
	}
}