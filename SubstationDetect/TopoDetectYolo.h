#pragma once
#include<opencv2/opencv.hpp>
#include"TopoDetect.h"
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum);
void SetLableBox(Mat& canva, Rect box, Scalar color, string text);
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
class TopoDetectYolo
{
public:
	//�мǣ�cross�������һλ��
	int classes;//�����˹�ϵ�е�Ԫ��������
	int objNum;//�����˹�ϵ�е���������,ʵ��Ϊͼ����ͼԪ+���������ܺ�
	Size canvaSize;
	Mat adjGraph;//����ͼ���䳤�Ϳ��ΪobjNum
	Mat adjGraphPure;//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
	vector<vector<Point>> objPixelVessel;//��������Ӧ���ؼ���vector<Point>��������
	vector<int> objTypeVessel;//��������Ӧ���͵�����
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
	void SetSize(Size in);
	void SetSize(int width, int height);
	void ResetGraph() {
		cout << "��"<< objPixelVessel.size() <<"��С����ͼ" << endl;
		adjGraph=Mat::eye(objPixelVessel.size(), objPixelVessel.size(),CV_32FC1);//����ͼ���䳤�Ϳ��ΪobjNum
		adjGraphPure=adjGraph.clone();//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
	}
	void DetectTopo();
	void ShowTopo(int showType);
};

