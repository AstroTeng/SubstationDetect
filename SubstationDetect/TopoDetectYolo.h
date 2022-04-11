#pragma once
#include<opencv2/opencv.hpp>
#include"TopoDetect.h"
#include "CoutShape.h"
#define itr_setint set<int>::iterator
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum);
void SetLableBox(Mat& canva, Rect box, Scalar color, string text);
vector<int> GetConnect(Mat adjMat, int  index);//����һ��Ӿ������ţ�����������ڸþ��������ӵ�������ŵ�����

const static vector<string> subEleNames = {
			"�����ѹ��",
			"�����ѹ��",
			"���뿪��",
			"��·��",
			"����",
			"�ӵ�",
			"�۶���",
			"������",
			"c_ud",
			"c_lr",
			"����"
};
const static vector<string> gapNames = {
	"�Ǽ��",
	#define GAP_NONE 0
	"��ͨ���",
	#define GAP_NORMAL 1
	"���߼��",
	#define GAP_ACROSS 2
	"��ͨ�������",
	#define GAP_NORMAL_OUT 3
	"���߼������",
	#define GAP_ACROSS_OUT 4
	"˫ĸ�ֶμ��",
	#define GAP_DBUS_SUBSECTION 5
	"��ĸ�ֶμ��"
	#define GAP_SBUS_SUBSECTION 6
};
const static vector<string> busNames = {
	"��ĸ��",
	#define BUS_NONE 0
	"ĸ��",
	#define BUS_ 1
	"��ĸ���ֶ�",
	#define BUS_SINGLE 2
	"˫ĸ���ֶ�",
	#define BUS_DOUBLE 3
	"��ĸ�ֶ�",
	#define BUS_SINGLE_SUB 4
	"˫ĸ�ֶ�"
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
	//�мǣ�cross�������һλ��
	int classes;//�����˹�ϵ�е�Ԫ��������
	int objNum;//�����˹�ϵ�е���������,ʵ��Ϊͼ����ͼԪ+���������ܺ�
	Size canvaSize;
	Mat adjGraph;//����ͼ���䳤�Ϳ��ΪobjNum
	Mat adjGraphPure;//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
	Mat adjGraphLine;//����������ʮ���ߣ�������Ԫ���͵���֮������ӹ�ϵ
	vector<vector<Point>> objPixelVessel;//��������Ӧ���ؼ���vector<Point>��������
	vector<int> objTypeVessel;//��������Ӧ���͵�����
	vector<int> gapTypeVessel;//�������������һ����������
	set<int> busNumbers;
	vector<Obj_gap> gap_s;
	vector<Obj_bus> bus_s;
	//0-�����ڼ��
	//1-��ͨ���
	//2-���߼��
	vector<int> gapNoVessel;//�������������һ�ż�������飬-1��ʾ��OBJ�����ڼ��
	vector<int> busTypeVessel;//�������������һ���ĸ��
	//0������ĸ��
	//1ĸ��Ԥ��
	//2��ĸ��
	//3˫ĸ��
	vector<int> busNoVessel;//�������������һ�ŵ�ĸ�ߣ�-1��ʾ�����岻Ϊĸ��
	TopoDetectYolo(int _classes) {
		classes = _classes;
		objNum = 0;
		canvaSize.width = 10;
		canvaSize.height = 10;
		adjGraph = Mat::eye(1, 1, CV_32FC1);//����ͼ���䳤�Ϳ��ΪobjNum
		adjGraphPure = adjGraph.clone();//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
	}
	//����type �� ��������ı�ǩ
	//0-classes-2����ͨͼԪ �� ��ǩ0-classes-2
	//classes-1-classes��cross������/���ұ� �� ��ǩclasses-1
	//classes+1���������� �� �޶�Ӧ��ǩ
	void AddObj(Rect in, int type);
	void AddObj(vector<Point> in);
	void SetSize(Size in);//���û�����С
	void SetSize(int width, int height);
	void ResetGraph() {
		cout << "��"<< objPixelVessel.size() <<"��С����ͼ" << endl;
		adjGraph=Mat::eye(objPixelVessel.size(), objPixelVessel.size(),CV_32FC1);//����ͼ���䳤�Ϳ��ΪobjNum
		adjGraphPure=adjGraph.clone();//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
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
	vector<int> GetAllLine(Mat adj,int index);//������Ӿ�����ߵ���ţ���������������������ߵ��������
	int NearestBus(Mat adj, int index);
	void ShowTopo(int showType);
	int ObjType(int index) {
		return objTypeVessel[index];
	}
private:
	Mat EliminateType(Mat inGraph, int type);
};

