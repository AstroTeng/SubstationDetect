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
		if (objTypeVessel[index] == classes - 1 || objTypeVessel[index] == classes) //先将十字绘制在画布上
		{
			//cout << "index:" << index << " objTypeVessel[index]:" << objTypeVessel[index] << endl;
				SetPoints(canva, objPixelVessel[index], index);
		}
	for (int index = 0; index < objPixelVessel.size(); index++)
		if (!(objTypeVessel[index] == classes - 1 || objTypeVessel[index] == classes)) {//再将其余图元线框绘制在画布上，同时检测碰撞
			vector<int> collisionObj = SetPoints(canva, objPixelVessel[index], index);
			for (int i = 0; i < collisionObj.size(); i++) {
				adjGraph.at<float>(index, collisionObj[i]) = 1;
				adjGraph.at<float>(collisionObj[i], index) = 1;
			}
		}

	adjGraphPure = adjGraph.clone();

	for (int m_type = classes + 1; m_type >= classes - 1; m_type--)
		for (int index = 0; index < objPixelVessel.size(); index++) {
			//找出原图中的所有导线和十字交叉线的上下/左右边沿，并将其相连的所有图元彼此相连，
			//再将该其和上述图元一一断开
			if (objTypeVessel[index] == m_type) {
				vector<int>connectElement;//用于记录和该导线相连的图元的号码
				for (int i = 0; i < objPixelVessel.size(); i++)
					if (adjGraphPure.at<float>(index, i) != 0)
						if (index != i) {
							connectElement.push_back(i);
							adjGraphPure.at<float>(index, i) = 0;
							adjGraphPure.at<float>(i, index) = 0;
						}
				//cout << "对于图中的" << index << "号" << subEleNames[m_type] << endl;
				if (connectElement.size() > 1)
					//若为0，则说明本连接元素不与任何图元相连
					//若为1，则说明本连接元素只和一个图元相连
					//以上两种情况都不用修改图元的连接关系
					for (int i = 0; i < connectElement.size() - 1; i++)
						for (int j = i + 1; j < connectElement.size(); j++) {
							adjGraphPure.at<float>(connectElement[i], connectElement[j]) = 1;
							adjGraphPure.at<float>(connectElement[j], connectElement[i]) = 1;
							//cout << "已将" << i << "号，" << j << "号连接" << endl;
						}
			}
		}
	cout << "拓扑检测完成！" << endl;
}
void TopoDetectYolo::ShowTopo(int showType) {
	switch (showType) {
	case TOPO_ALL:
		for (int index = 0; index < adjGraph.rows; index++) {
			cout << index << "号" << subEleNames[objTypeVessel[index]] << " → ";
			for (int i = 0; i < adjGraph.cols; i++)
				if (adjGraph.at<float>(index, i) != 0)
					if (index != i)
						cout << i << "号" << subEleNames[objTypeVessel[i]] << " ";
			cout << endl;
		}
		break;
	case TOPO_LINE_EXCEPT:
		for (int index = 0; index < adjGraphPure.rows; index++)
			if (objTypeVessel[index] != classes-1 && objTypeVessel[index] != classes  && objTypeVessel[index] != classes + 1) {
				cout << index << "号" << subEleNames[objTypeVessel[index]] << " → ";
				for (int i = 0; i < adjGraphPure.cols; i++)
					if (adjGraphPure.at<float>(index, i) != 0)
						if (index != i)
							cout << i << "号" << subEleNames[objTypeVessel[i]] << " ";
				cout << endl;
			}
		break;
	}
}
vector<int> SetPoints(Mat& canva, vector<Point> p2draw, int objNum) {
	vector<int> collisionObjVesssel;//返回的是在绘制过程中碰撞的OBJ序号
	collisionObjVesssel.push_back(objNum);
	if (canva.type() != CV_32FC1)
		cout << "错误！画布的类型不为CV_32FC1" << endl;
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
	collisionObjVesssel.assign(s.begin(), s.end());//消除重复



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