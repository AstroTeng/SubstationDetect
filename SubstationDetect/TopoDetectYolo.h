#pragma once
#include<opencv2/opencv.hpp>
#include"TopoDetect.h"
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum);
const static vector<string> subEleNames = {
			"三相变压器",
			"两相变压器",
			"隔离开关",
			"断路器",
			"负载",
			"接地",
			"熔断器",
			"避雷器",
			"c_ud",
			"c_lr",
			"导线"
};
class TopoDetectYolo
{
public:
	//切记：cross放在最后一位！
	int classes;//该拓扑关系中的元素种类数
	int objNum;//该拓扑关系中的物体数量,实际为图像中图元+导线数的总和
	Size canvaSize;
	Mat adjGraph;//无向图，其长和宽皆为objNum
	Mat adjGraphPure;//忽略了导线和十字线，仅仅体现了各个元件之间关系的无向图
	vector<vector<Point>> objPixelVessel;//存放物体对应像素集（vector<Point>）的容器
	vector<int> objTypeVessel;//存放物体对应类型的容器
	TopoDetectYolo(int _classes) {
		classes = _classes;
		objNum = 0;
		canvaSize.width = 10;
		canvaSize.height = 10;
		adjGraph = Mat::eye(1, 1, CV_32FC1);//无向图，其长和宽皆为objNum
		adjGraphPure = adjGraph.clone();//忽略了导线和十字线，仅仅体现了各个元件之间关系的无向图
	}
	//类型type ← 网络检测出的标签
	//0-classes-2：普通图元 ← 标签0-classes-2
	//classes-1-classes：cross的上下/左右边 ← 标签classes-1
	//classes+1：自由曲线 ← 无对应标签
	void AddObj(Rect in, int type);
	void AddObj(vector<Point> in);
	void SetSize(Size in);
	void SetSize(int width, int height);
	void ResetGraph() {
		cout << "以"<< objPixelVessel.size() <<"大小重置图" << endl;
		adjGraph=Mat::eye(objPixelVessel.size(), objPixelVessel.size(),CV_32FC1);//无向图，其长和宽皆为objNum
		adjGraphPure=adjGraph.clone();//忽略了导线和十字线，仅仅体现了各个元件之间关系的无向图
	}
	void DetectTopo();
	void ShowTopo(int showType);
};

