#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
vector< vector<Point>> RectFetchEdge(Rect in);
vector<int> SetPointOn(Mat &canva,vector<Point> p2draw,int objNum);
Mat Mat2Pic(Mat in);
class Graph {
public:
	int objNum;
	Mat adjMat;//领接矩阵
	Graph() {
		objNum = 1;
		adjMat = Mat::eye(objNum, objNum, CV_32FC1);
	}
	Graph(int length) {
		objNum = length;
		adjMat = Mat::eye(objNum, objNum, CV_32FC1);
	}
    Graph operator = (Graph graph_r){
        this->objNum=graph_r.objNum;
        this->adjMat=graph_r.adjMat.clone();
		return *this;
    }
	void Reset(int length) {
		objNum = length;
		adjMat = Mat::eye(objNum, objNum, CV_32FC1);
	}
	void ConnetUndir(int obj1, int obj2) {
		adjMat.at<float>(obj1, obj2) = 1;
		adjMat.at<float>(obj2, obj1) = 1;
	}
	void DisConnetUndir(int obj1, int obj2) {
		adjMat.at<float>(obj1, obj2) = 0;
		adjMat.at<float>(obj2, obj1) = 0;
	}
	void ConnetDir(int obj_from, int obj_to) {
		adjMat.at<float>(obj_from, obj_to) = 1;
	}
	void DisConnetDir(int obj_from, int obj_to) {
		adjMat.at<float>(obj_from, obj_to) = 0;

	}
    void ShowGraph(){
        cout<<adjMat<<endl;
    }
    void ShowConnect(int obj2show){
        cout<<"图中的"<<obj2show<<"号顶点与图中下述其它顶点相连："<<endl;
        for(int index=0;index<objNum;index++)
            if(index!=obj2show)
                if(adjMat.at<float>(obj2show,index)!=0)
                    cout<<index<<" ";
        cout<<endl;
    }
};
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
		objNum = 1;
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

