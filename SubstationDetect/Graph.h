#pragma once
#include<opencv2\opencv.hpp>
using namespace cv;
using namespace std;
class Graph {
public:
	int objNum;
	Mat adjMat;//领接矩阵
	Graph() {
		objNum = 1;
		adjMat = Mat::eye(objNum, objNum, CV_64FC1);
	}
	Graph(int length) {
		objNum = length;
		adjMat = Mat::eye(objNum, objNum, CV_64FC1);
	}
	Graph operator = (Graph graph_r) {
		this->objNum = graph_r.objNum;
		this->adjMat = graph_r.adjMat.clone();
		return *this;
	}
	void Reset(int length) {
		objNum = length;
		adjMat = Mat::eye(objNum, objNum, CV_64FC1);
	}
	void ConnetUndir(int obj1, int obj2) {
		adjMat.at<double>(obj1, obj2) = 1;
		adjMat.at<double>(obj2, obj1) = 1;
	}
	void DisConnetUndir(int obj1, int obj2) {
		adjMat.at<double>(obj1, obj2) = 0;
		adjMat.at<double>(obj2, obj1) = 0;
	}
	void ConnetDir(int obj_from, int obj_to) {
		adjMat.at<double>(obj_from, obj_to) = 1;
	}
	void DisConnetDir(int obj_from, int obj_to) {
		adjMat.at<double>(obj_from, obj_to) = 0;

	}
	void ShowGraph() {
		cout << adjMat << endl;
	}
	void ShowConnect(int obj2show) {
		cout << "图中的" << obj2show << "号顶点与图中下述其它顶点相连：" << endl;
		for (int index = 0; index < objNum; index++)
			if (index != obj2show)
				if (adjMat.at<double>(obj2show, index) != 0)
					cout << index << " ";
		cout << endl;
	}
};

