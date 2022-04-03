#include "TopoDetectYolo.h"
void TopoDetectYolo::AddObj(Rect in, int type) {
	vector<vector<Point>> rectEdges = RectFetchEdge(in);
	vector<Point> ud;
	vector<Point> lr;
	vector<Point> tmp;
	if (type == classes - 1) {
		ud = rectEdges[0];
		ud.insert(ud.end(), rectEdges[2].begin(), rectEdges[2].end());
		objPixelVessel.push_back(ud);
		objTypeVessel.push_back(classes - 1);
		objNum++;
		lr = rectEdges[1];
		lr.insert(lr.end(), rectEdges[3].begin(), rectEdges[3].end());
		objPixelVessel.push_back(lr);
		objTypeVessel.push_back(classes);
		objNum++;
	}
	else if (type != classes + 1) {
		tmp = rectEdges[0];
		tmp.insert(tmp.end(), rectEdges[1].begin(), rectEdges[1].end());
		tmp.insert(tmp.end(), rectEdges[2].begin(), rectEdges[2].end());
		tmp.insert(tmp.end(), rectEdges[3].begin(), rectEdges[3].end());
		objPixelVessel.push_back(tmp);
		objTypeVessel.push_back(type);
		objNum++;
	}
	
}
void TopoDetectYolo::AddObj(vector<Point> in) {
	objNum++;
	objPixelVessel.push_back(in);
	objTypeVessel.push_back(classes + 1);
}
void TopoDetectYolo::SetSize(Size in) {
	canvaSize = in;
}
void TopoDetectYolo::SetSize(int width, int height) {
	canvaSize.width = width;
	canvaSize.height = height;
}
void TopoDetectYolo::DetectTopo() {
	Mat canva = Mat::zeros(canvaSize, CV_32FC1);
	canva -= 1;
	for (int index = 0; index < objPixelVessel.size(); index++)
		if (objTypeVessel[index] == classes - 1 || objTypeVessel[index] == classes) //�Ƚ�ʮ�ֻ����ڻ�����
		{
			//cout << "index:" << index << " objTypeVessel[index]:" << objTypeVessel[index] << endl;
				SetPoints(canva, objPixelVessel[index], index);
		}
	for (int index = 0; index < objPixelVessel.size(); index++)
		if (!(objTypeVessel[index] == classes - 1 || objTypeVessel[index] == classes)) {//�ٽ�����ͼԪ�߿�����ڻ����ϣ�ͬʱ�����ײ
			vector<int> collisionObj = SetPoints(canva, objPixelVessel[index], index);
			for (int i = 0; i < collisionObj.size(); i++) {
				adjGraph.at<float>(index, collisionObj[i]) = 1;
				adjGraph.at<float>(collisionObj[i], index) = 1;
			}
		}

	adjGraphPure = adjGraph.clone();

	for (int m_type = classes + 1; m_type >= classes - 1; m_type--)
		for (int index = 0; index < objPixelVessel.size(); index++) {
			//�ҳ�ԭͼ�е����е��ߺ�ʮ�ֽ����ߵ�����/���ұ��أ�����������������ͼԪ�˴�������
			//�ٽ����������ͼԪһһ�Ͽ�
			if (objTypeVessel[index] == m_type) {
				vector<int>connectElement;//���ڼ�¼�͸õ���������ͼԪ�ĺ���
				for (int i = 0; i < objPixelVessel.size(); i++)
					if (adjGraphPure.at<float>(index, i) != 0)
						if (index != i) {
							connectElement.push_back(i);
							adjGraphPure.at<float>(index, i) = 0;
							adjGraphPure.at<float>(i, index) = 0;
						}
				//cout << "����ͼ�е�" << index << "��" << subEleNames[m_type] << endl;
				if (connectElement.size() > 1)
					//��Ϊ0����˵��������Ԫ�ز����κ�ͼԪ����
					//��Ϊ1����˵��������Ԫ��ֻ��һ��ͼԪ����
					//������������������޸�ͼԪ�����ӹ�ϵ
					for (int i = 0; i < connectElement.size() - 1; i++)
						for (int j = i + 1; j < connectElement.size(); j++) {
							adjGraphPure.at<float>(connectElement[i], connectElement[j]) = 1;
							adjGraphPure.at<float>(connectElement[j], connectElement[i]) = 1;
							//cout << "�ѽ�" << i << "�ţ�" << j << "������" << endl;
						}
			}
		}
	cout << "���˼����ɣ�" << endl;
}
void TopoDetectYolo::ShowTopo(int showType) {
	switch (showType) {
	case TOPO_ALL:
		for (int index = 0; index < adjGraph.rows; index++) {
			cout << index << "��" << subEleNames[objTypeVessel[index]] << " �� ";
			for (int i = 0; i < adjGraph.cols; i++)
				if (adjGraph.at<float>(index, i) != 0)
					if (index != i)
						cout << i << "��" << subEleNames[objTypeVessel[i]] << " ";
			cout << endl;
		}
		break;
	case TOPO_LINE_EXCEPT:
		for (int index = 0; index < adjGraphPure.rows; index++)
			if (objTypeVessel[index] != classes-1 && objTypeVessel[index] != classes  && objTypeVessel[index] != classes + 1) {
				cout << index << "��" << subEleNames[objTypeVessel[index]] << " �� ";
				for (int i = 0; i < adjGraphPure.cols; i++)
					if (adjGraphPure.at<float>(index, i) != 0)
						if (index != i)
							cout << i << "��" << subEleNames[objTypeVessel[i]] << " ";
				cout << endl;
			}
		break;
	}
}
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum) {
	vector<int> collisionObjVesssel;//���ص����ڻ��ƹ�������ײ��OBJ���
	collisionObjVesssel.push_back(objNum);
	if (canva.type() != CV_32FC1)
		cout << "���󣡻��������Ͳ�ΪCV_32FC1" << endl;
	for (int i = 0; i < p2draw.size(); i++) {
		int x = p2draw[i].x;
		int y = p2draw[i].y;
		//cout << "at " << y << "," << x << " (int)canva.at<float>(y, x)=" << (int)canva.at<float>(y, x) << endl;
		if (canva.at<float>(y, x) != -1) {
			collisionObjVesssel.push_back((int)canva.at<float>(y, x));
			canva.at<float>(y, x) = -1;
		}
		else {
			//collisionObjVesssel.push_back(-1);
			canva.at<float>(y, x) = objNum;
		}
	}


	set<int> s(collisionObjVesssel.begin(), collisionObjVesssel.end());
	collisionObjVesssel.assign(s.begin(), s.end());//�����ظ�



	return collisionObjVesssel;
}
void SetLableBox(Mat& canva, Rect box, Scalar color, string text) {
	rectangle(canva, box, color, 1);
	Rect tiny_box = box;
	tiny_box.y -= 10;
	rectangle(canva, box, color, 1);
	rectangle(canva, tiny_box, color, -1);
	putText(canva, text, Point(box.x, box.y), FONT_HERSHEY_COMPLEX_SMALL, 0.3, Scalar(255, 255, 255), 1);
}