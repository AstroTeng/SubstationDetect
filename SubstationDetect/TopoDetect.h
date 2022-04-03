#pragma once
#include <opencv2/opencv.hpp>
#include"Graph.h"
using namespace cv;
using namespace std;
vector< vector<Point>> RectFetchEdge(Rect in);
vector<int> SetPointOn(Mat &canva,vector<Point> p2draw,int objNum);
Mat Mat2Pic(Mat in);

class TopoDetect
{
public:
	int objNum;//该拓扑关系中的物体数量,实际为图像中图元+导线数的总和
	Size canvaSize;
	Graph graph;//无向图，其长和宽皆为objNum
    Graph graphPure;//忽略了导线和十字线，仅仅体现了各个元件之间关系的无向图
	vector<vector<Point>> objPixelVessel;//存放物体对应像素集（vector<Point>）的容器
	vector<int> objTypeVessel;//存放物体对应类型的容器
	//0-5：普通图元 ← 标签0-5
	//6-7：cross的上下/左右边 ← 标签6
	//无对应： ← 标签7
	//8：自由曲线 ← 无对应标签
	TopoDetect() {
		objNum = 0;
		canvaSize.width = 10;
		canvaSize.height = 10;
	}
	void AddObj(Rect in, int type);
	void AddObj(vector<Point> in);
	void SetSize(Size in);
	void SetSize(int width, int height);
	void ShowInPic();
	void ResetGraph();
	void DetectTopo();

#define TOPO_ALL 0
#define TOPO_LINE_EXCEPT 1
    void ShowTopo(int showType);

};

