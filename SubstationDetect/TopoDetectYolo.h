#pragma once
#include<opencv2/opencv.hpp>
#include"TopoDetect.h"
#include "CoutShape.h"
#define itr_setint set<int>::iterator
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum);
void SetLableBox(Mat& canva, Rect box, Scalar color, string text);
vector<int> GetConnect(Mat adjMat, int  index);//给出一领接矩阵和序号，给出该序号在该矩阵中连接的其它序号的数组

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
const static vector<string> gapNames = {
	"非间隔",
	#define GAP_NONE 0
	"普通间隔",
	#define GAP_NORMAL 1
	"跨线间隔",
	#define GAP_ACROSS 2
	"普通间隔出线",
	#define GAP_NORMAL_OUT 3
	"跨线间隔出线",
	#define GAP_ACROSS_OUT 4
	"双母分段间隔",
	#define GAP_DBUS_SUBSECTION 5
	"单母分段间隔"
	#define GAP_SBUS_SUBSECTION 6
};
const static vector<string> busNames = {
	"非母线",
	#define BUS_NONE 0
	"母线",
	#define BUS_ 1
	"单母不分段",
	#define BUS_SINGLE 2
	"双母不分段",
	#define BUS_DOUBLE 3
	"单母分段",
	#define BUS_SINGLE_SUB 4
	"双母分段"
	#define BUS_DOUBLE_SUB 5
};
const static vector<Scalar> subEleColors = {
			Scalar(51,51,255),
			Scalar(0,128,255),
			Scalar(255,128,0),
			Scalar(204,0,102),
			Scalar(0,204,102),
			Scalar(102,0,204),
			Scalar(0,204,204),
			Scalar(127,0,255),
			Scalar(64,64,64),
			Scalar(64,64,64),
			Scalar(64,64,64)
};
class Obj {
public:
	Obj() {
		type = 0;
	}
	int type;
	set<int> obj_indexs;
	void Insert(int index) {
		obj_indexs.insert(index);
	}
	void Insert(Obj _obj) {
		for (itr_setint itr = _obj.obj_indexs.begin(); itr != _obj.obj_indexs.end(); itr++) {
			obj_indexs.insert(*itr);
		}
	}
	void ShowObjs() {
		for (itr_setint itr = obj_indexs.begin(); itr != obj_indexs.end(); itr++)
			cout << *itr << ",";
		cout << endl;
	}
};
class Obj_bus:public Obj {

};
class Obj_gap :public Obj {
public:
	set<int> SideObjs(Mat adjGraph);
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
	Mat adjGraphLine;//仅仅忽略了十字线，保留了元件和导线之间的连接关系
	vector<vector<Point>> objPixelVessel;//存放物体对应像素集（vector<Point>）的容器
	vector<int> objTypeVessel;//存放物体对应类型的容器
	vector<int> gapTypeVessel;//存放物体属于哪一类间隔的数组
	set<int> busNumbers;
	vector<Obj_gap> gap_s;
	vector<Obj_bus> bus_s;
	//0-不属于间隔
	//1-普通间隔
	//2-跨线间隔
	vector<int> gapNoVessel;//存放物体属于哪一号间隔的数组，-1表示该OBJ不属于间隔
	vector<int> busTypeVessel;//存放物体属于那一类的母线
	//0不属于母线
	//1母线预备
	//2单母线
	//3双母线
	vector<int> busNoVessel;//存放物体属于哪一号的母线，-1表示该物体不为母线
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
	void SetSize(Size in);//设置画布大小
	void SetSize(int width, int height);
	void ResetGraph() {
		cout << "以"<< objPixelVessel.size() <<"大小重置图" << endl;
		adjGraph=Mat::eye(objPixelVessel.size(), objPixelVessel.size(),CV_32FC1);//无向图，其长和宽皆为objNum
		adjGraphPure=adjGraph.clone();//忽略了导线和十字线，仅仅体现了各个元件之间关系的无向图
		gapTypeVessel = gapNoVessel = objTypeVessel;
		for (int i = 0; i < gapTypeVessel.size(); i++) {
			gapTypeVessel[i] = 0;
			gapNoVessel[i] = -1;
		}
		busTypeVessel = gapTypeVessel;
		busNoVessel = gapNoVessel;
			
	}
	void DetectTopo();
	void DetectGapBus();
	void DetectGapBus2();
	vector<int> GetAllLine(Mat adj,int index);//给出领接矩阵和线的序号，给出与该线相连的所有线的序号数组
	int NearestBus(Mat adj, int index);
	void ShowTopo(int showType);
	int ObjType(int index) {
		return objTypeVessel[index];
	}
private:
	Mat EliminateType(Mat inGraph, int type);
};

