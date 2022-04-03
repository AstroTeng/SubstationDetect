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
	int objNum;//�����˹�ϵ�е���������,ʵ��Ϊͼ����ͼԪ+���������ܺ�
	Size canvaSize;
	Graph graph;//����ͼ���䳤�Ϳ��ΪobjNum
    Graph graphPure;//�����˵��ߺ�ʮ���ߣ����������˸���Ԫ��֮���ϵ������ͼ
	vector<vector<Point>> objPixelVessel;//��������Ӧ���ؼ���vector<Point>��������
	vector<int> objTypeVessel;//��������Ӧ���͵�����
	//0-5����ͨͼԪ �� ��ǩ0-5
	//6-7��cross������/���ұ� �� ��ǩ6
	//�޶�Ӧ�� �� ��ǩ7
	//8���������� �� �޶�Ӧ��ǩ
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

