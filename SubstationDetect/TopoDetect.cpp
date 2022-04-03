#include "TopoDetect.h"
Mat Mat2Pic(Mat in) {
	Mat out(in.rows, in.cols, CV_8UC1, Scalar(255));
	for (int m = 0; m < in.rows; m++)
		for (int n = 0; n < in.cols; n++)
			if (in.at<float>(m, n) == 1)
				out.at<uchar>(m, n) = 0;
	resize(out, out, Size(0, 0), 4, 4, INTER_NEAREST);
	return out;
}
void TopoDetect::AddObj(Rect in, int type) {
	vector<vector<Point>> rectEdges = RectFetchEdge(in);
	vector<Point> ud;
	vector<Point> lr;
	vector<Point> tmp;
	switch (type) {
	case 7:break;//7Ϊ���հף�����Ҫ����
	case 6://6��ΪCROSS����Ҫ�������д���
		ud = rectEdges[0];
		ud.insert(ud.end(), rectEdges[2].begin(), rectEdges[2].end());
		objPixelVessel.push_back(ud);
		objTypeVessel.push_back(6);
		objNum++;
		lr = rectEdges[1];
		lr.insert(lr.end(), rectEdges[3].begin(), rectEdges[3].end());
		objPixelVessel.push_back(lr);
		objTypeVessel.push_back(7);
		objNum++;
		break;
	default:
		tmp = rectEdges[0];
		tmp.insert(tmp.end(), rectEdges[1].begin(), rectEdges[1].end());
		tmp.insert(tmp.end(), rectEdges[2].begin(), rectEdges[2].end());
		tmp.insert(tmp.end(), rectEdges[3].begin(), rectEdges[3].end());
		objPixelVessel.push_back(tmp);
		objTypeVessel.push_back(type);
		objNum++;
		break;
	}
}
void TopoDetect::AddObj(vector<Point> in) {
	objNum++;
	objPixelVessel.push_back(in);
	objTypeVessel.push_back(8);
}
void TopoDetect::SetSize(Size in) {
	canvaSize = in;
}
void TopoDetect::SetSize(int width, int height) {
	canvaSize.width = width;
	canvaSize.height = height;
}
void TopoDetect::ShowInPic() {
	Mat canva = Mat(canvaSize.height, canvaSize.width, CV_8UC3, Scalar(255, 255, 255));

	for (int index = 0; index <objPixelVessel.size(); index++) {
		Scalar color(0, 50, 100);
		if (objTypeVessel[index] == 6)
			color = Scalar(255, 0, 0);
		if (objTypeVessel[index] == 7)
			color = Scalar(0, 0, 255);
		drawContours(canva, objPixelVessel, index, color, 1);
        //P.S.�ú����ڻ���ʱ���Զ����Ӽ�ϵ�point

	}
	for (int index = 0; index < objPixelVessel.size(); index++) {
		putText(canva,
			to_string(index) + ":" + to_string(objTypeVessel[index]),
			objPixelVessel[index][0],
			FONT_HERSHEY_DUPLEX, 0.25, Scalar(0, 0, 0), 1);
	}
	imshow("TOPO", canva);
	waitKey(10);
}
void TopoDetect::ResetGraph() {
	graph.Reset(objPixelVessel.size());
}
void TopoDetect::DetectTopo() {
	Mat canva = Mat::zeros(canvaSize, CV_32FC1);
	canva -= 1;
	for (int index = 0; index < objPixelVessel.size(); index++)
		if (objTypeVessel[index] == 6 || objTypeVessel[index] == 7) //�Ƚ�ʮ�ֻ����ڻ�����
			SetPointOn(canva, objPixelVessel[index], index);

    for(int index=0;index<objPixelVessel.size();index++)
        if(!(objTypeVessel[index] == 6 || objTypeVessel[index] == 7)){//�ٽ�����ͼԪ�߿�����ڻ����ϣ�ͬʱ�����ײ
			vector<int> collisionObj = SetPointOn(canva, objPixelVessel[index], index);
            for(int i=0;i<collisionObj.size();i++)
                this->graph.ConnetUndir(index,collisionObj[i]);
        }
    graphPure=graph;
    for(int m_type=8;m_type>=6;m_type--)
        for(int index=0;index<objPixelVessel.size();index++)
			//�ҳ�ԭͼ�е����е��ߺ�ʮ�ֽ����ߵ�����/���ұ��أ�����������������ͼԪ�˴�������
            //�ٽ����������ͼԪһһ�Ͽ�
            if(objTypeVessel[index]==m_type){
                vector<int>connectElement;//���ڼ�¼�͸õ���������ͼԪ�ĺ���
                for(int i=0;i<graphPure.objNum;i++)
                    if(graphPure.adjMat.at<float>(index,i)!=0)
                        if(index!=i){
                            connectElement.push_back(i);
                            graphPure.DisConnetUndir(index,i);
                        }
                if(connectElement.size()>1)
                    //��Ϊ0����˵��������Ԫ�ز����κ�ͼԪ����
                    //��Ϊ1����˵��������Ԫ��ֻ��һ��ͼԪ����
                    //������������������޸�ͼԪ�����ӹ�ϵ
                    for(int i=0;i<connectElement.size()-1;i++)
                        for(int j=i+1;j<connectElement.size();j++)
                            graphPure.ConnetUndir(connectElement[i],connectElement[j]);
            }
    cout<<"���˼����ɣ�"<<endl;
}
void TopoDetect::ShowTopo(int showType){
    vector<string> names={
            "�����ѹ��",
            "�����ѹ��",
            "���뿪��",
            "��·��",
            "����",
            "�ӵ�",
            "������������",
            "������������",
            "����"
    };
    switch (showType) {
        case TOPO_ALL:
            for(int index=0;index<graph.adjMat.rows;index++) {
                cout << index << "��" << names[objTypeVessel[index]] << " �� ";
                for (int i = 0; i < graph.adjMat.cols; i++)
                    if (graph.adjMat.at<float>(index, i) != 0)
                        if (index != i)
                            cout << i << "��" << names[objTypeVessel[i]] << " ";
                cout << endl;
            }
            break;
        case TOPO_LINE_EXCEPT:
            for(int index=0;index<graphPure.adjMat.rows;index++)
                if(objTypeVessel[index]!=6&&objTypeVessel[index]!=7&&objTypeVessel[index]!=8){
                    cout << index << "��" << names[objTypeVessel[index]] << " �� ";
                    for (int i = 0; i < graphPure.adjMat.cols; i++)
                        if (graphPure.adjMat.at<float>(index, i) != 0)
                            if (index != i)
                                cout << i << "��" << names[objTypeVessel[i]] << " ";
                    cout << endl;
                }
            break;
    }
}
vector< vector<Point>> RectFetchEdge(Rect in) {
	vector< vector<Point>> out;
	vector<Point> tmp1;
	for (int x = in.x; x < in.x + in.width - 1; x++)
		tmp1.push_back(Point(x, in.y));
	out.push_back(tmp1);
	vector<Point> tmp2;
	for (int y = in.y; y < in.y + in.height - 1; y++)
		tmp2.push_back(Point(in.x + in.width - 1, y));
	out.push_back(tmp2);
	vector<Point> tmp3;
	for (int x = in.x + in.width - 1; x > in.x; x--)
		tmp3.push_back(Point(x, in.y + in.height - 1));
	out.push_back(tmp3);
	vector<Point> tmp4;
	for (int y = in.y + in.height - 1; y > in.y; y--)
		tmp4.push_back(Point(in.x, y));
	out.push_back(tmp4);
	return out;
}

vector<int> SetPointOn(Mat &canva,vector<Point> p2draw,int objNum){
    vector<int> collisionObjVesssel;//���ص����ڻ��ƹ�������ײ��OBJ���
	collisionObjVesssel.push_back(objNum);
    if(canva.type()!=CV_32FC1)
        cout<<"���󣡻��������Ͳ�ΪCV_32FC1"<<endl;
    for(int i=0;i<p2draw.size();i++){
        int x=p2draw[i].x;
        int y=p2draw[i].y;
        if(canva.at<float>(y,x)!=-1){
            collisionObjVesssel.push_back(canva.at<float>(y,x));
            canva.at<float>(y,x)=-1;
        }
		else { 
			//collisionObjVesssel.push_back(-1);
			canva.at<float>(y, x) = objNum; 
		}
    }

    set<int> s(collisionObjVesssel.begin(),collisionObjVesssel.end());
    collisionObjVesssel.assign(s.begin(), s.end());//�����ظ�

	for (int i = 0; i < collisionObjVesssel.size(); i++)
		cout <<"cc"<< collisionObjVesssel[i] << endl;

    return collisionObjVesssel;
}