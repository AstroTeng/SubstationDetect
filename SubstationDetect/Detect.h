#pragma once
#include "Matrix.h"
#include "ElementGenerate.h"
#include "CoutShape.h"
#include <random>
#include <fstream>
Mat DisWhiteBorder(Mat in);
void LableUp(String path);
Mat MatAnd(Mat src, Mat mask);
#define SEG_HEIGHT 50
#define SEG_WIDTH 50
class Segmentor {
public:
	String name;
	Mat src;
	Mat tuned;
	int seg_h;//纵向分块高度
	int seg_w;//横向分块宽度
	int rows;
	int cols;
	String savePath;
	Segmentor(int seg_h, int seg_w, String savePath,String name) {
		this->seg_w = seg_w;
		this->seg_h = seg_h;
		this->savePath = savePath;
		this->name = name;
	}
	void LoadImg(String path,bool if_cutwhite,bool if_binary) {
		src = imread(path);
		if (if_cutwhite) 		src = DisWhiteBorder(src);
		if (if_binary) 		threshold(src, src, 200, 255, THRESH_BINARY);
		cout << "原尺寸：" << src.rows << "×" << src.cols << endl;
		tuned = src.clone();
		resize(tuned, tuned, Size(src.cols+ seg_w - src.cols % seg_w, src.rows + seg_h - src.rows % seg_h));
		cout << "扩充尺寸：" << tuned.rows << "×" << tuned.cols << endl;
		rows = tuned.rows / seg_h;
		cols = tuned.cols / seg_w;
	}
	void ShowSeg() {
		Mat canva = tuned.clone();
		for (int i = seg_h; i < canva.rows; i += seg_h) {
			line(canva, Point(0,i), Point(canva.cols,i), Scalar(200), 1);
		}
		for (int j = seg_w; j < canva.cols; j += seg_w) {
			line(canva, Point(j, 0), Point(j, canva.cols), Scalar(200), 1);
		}
		//imshow("9", canva);
		imwrite(savePath + name + "seg.jpg", canva);
		//waitKey(0);
	}
	void SegUp() {
		int index = 0;
		for(int y=0;y* seg_h <tuned.rows;y++)
			for (int x = 0; x * seg_w < tuned.cols; x++) {
				Rect seg(x * seg_w, y * seg_h, seg_w, seg_h);
				String g_name = name;
				g_name = g_name  + to_string(y) + "-" + to_string(x) + ".jpg";
				g_name = to_string(10000 + index)+"-" + g_name;
				g_name[0] = 'n';
				index++;
				imwrite(savePath + g_name, tuned(seg));
			}			
	}
	Rect SegRect(int m, int n) {
		if (m < 0) m = 0;
		if (n < 0)n = 0;
		if (m >= rows) m = rows - 1;
		if (n >= cols) n = cols - 1;
		Rect out(n * seg_w, m * seg_h, seg_w, seg_h);
		return out;
	}
};
