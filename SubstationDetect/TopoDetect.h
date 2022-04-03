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
	Mat adjMat;//��Ӿ���
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
        cout<<"ͼ�е�"<<obj2show<<"�Ŷ�����ͼ��������������������"<<endl;
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

