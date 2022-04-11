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
	adjGraphLine = adjGraph.clone();
	for (int m_type = classes + 1; m_type >= classes - 1; m_type--)
		adjGraphPure = EliminateType(adjGraphPure, m_type);
	for (int m_type = classes; m_type >= classes - 1; m_type--)
		adjGraphLine = EliminateType(adjGraphLine, m_type);
	
	cout << "���˼����ɣ�" << endl;
}
void TopoDetectYolo::DetectGapBus() {
	//�����Ǽ����ⲿ��
	cout << "���м����⡭��";
	int gapCount = 0;
	for (int index = 0; index < objTypeVessel.size(); index++) {
		if (objTypeVessel[index] == 3) {//�ȶ�λ�����ж�·��
			vector<int> connect_what = GetConnect(adjGraphPure, index);
			if (connect_what.size() > 0) {
				vector<int> connectTypeCount(classes + 2);
				for (int i = 0; i < connect_what.size(); i++)
					connectTypeCount[objTypeVessel[connect_what[i]]]++;//������ö�·����ϵ������Ԫ����
				if (connectTypeCount[2] == 2) {
					for (int i = 0; i < connect_what.size(); i++) {
						gapTypeVessel[connect_what[i]] = GAP_NORMAL;
						gapNoVessel[connect_what[i]] = gapCount;
					}
					gapTypeVessel[index] = 1;
				}
				else if (connectTypeCount[2] == 3) {
					for (int i = 0; i < connect_what.size(); i++) {
						gapTypeVessel[connect_what[i]] = GAP_ACROSS;
						gapNoVessel[connect_what[i]] = gapCount;
					}
					gapTypeVessel[index] = 2;
				}
				else if (connectTypeCount[2] == 4) {
					for (int i = 0; i < connect_what.size(); i++) {
						gapTypeVessel[connect_what[i]] = GAP_DBUS_SUBSECTION;
						gapNoVessel[connect_what[i]] = gapCount;
					}
					gapTypeVessel[index] = 5;
				}
				gapNoVessel[index] = gapCount;
				gapCount++;
			}
		}
	}
	cout << "���" << endl;
	//for (int n = 0; n < gapCount; n++) {//�����ʾ
	//	int g_type = 0;
	//	for (int index = 0; index < gapNoVessel.size(); index++) 
	//		if (gapNoVessel[index] == n) {
	//			cout << index << "��[" << subEleNames[objTypeVessel[index]] << "],";
	//			g_type = gapTypeVessel[index];
	//		}
	//	cout << endl << "���" << n << "��" << gapNames[g_type] << endl;
	//}
	//������ĸ�߳�����ⲿ��
	cout << "����ĸ�߳�����⡭��";
	int busCount = 0;
	for (int index = 0; index < busTypeVessel.size(); index++) {
		if (ObjType(index) != classes + 1)continue;
		if (busTypeVessel[index] == 1)continue;
		vector<int> allLine = GetAllLine(this->adjGraphLine, index);
		//cout << "����" << index << "�źͣ�";
		vector<int> c_type_count(classes + 2);
		for (int i = 0; i < allLine.size(); i++) {
			//cout << allLine[i] << ",";
			vector<int> c_what = GetConnect(adjGraphLine, allLine[i]);
			for (int j = 0; j < c_what.size(); j++)
				c_type_count[objTypeVessel[c_what[j]]]++;
		}
		//cout << endl;
		bool not_a_bus = false;
		if (c_type_count[2] < 2) not_a_bus = true;
		for (int type = 0; type < classes + 1; type++) {
			if (type == 2)continue;
			if (c_type_count[type] > 0) not_a_bus = true;
		}
		if (not_a_bus) continue;
		for (int i = 0; i < allLine.size(); i++) {
			busTypeVessel[allLine[i]] = BUS_;
			busNoVessel[allLine[i]] = busCount;
		}
		busCount++;

	}
	cout << "���" << endl;
	//������ĸ�߾��������ⲿ��
	cout << "����ĸ�߷��࡭��" ;
	for (int n = 0; n < busCount; n++) {
		vector<int> g_type_count(3);//�뵱ǰn��ĸ�������ļ���������
		for (int index = 0; index < busNoVessel.size(); index++)
			if (n == busNoVessel[index]) {
				vector<int> c_what = GetConnect(adjGraphLine, index);
				for (int i = 0; i < c_what.size(); i++)
					g_type_count[gapTypeVessel[c_what[i]]]++;
			}
		int newtype;
		//cout << "g_type_count[1]" << g_type_count[1] << ",g_type_count[2]" << g_type_count[2] << endl;
		if (g_type_count[1] > g_type_count[2])  newtype = BUS_SINGLE; 
		if (g_type_count[1] <= g_type_count[2]) newtype = BUS_DOUBLE;
		for (int index = 0; index < busNoVessel.size(); index++)
			if (n == busNoVessel[index])
				busTypeVessel[index] = newtype;
	}
	cout << "���" << endl;
	//for (int n = 0; n < busCount; n++) {
	//	int bustype = 0;
	//	for (int index = 0; index < busNoVessel.size(); index++)
	//		if (n == busNoVessel[index]) {
	//			//cout << index << "�ŵ��ߣ�";
	//			bustype = busTypeVessel[index];
	//			//cout << "bustype" << bustype << "index"<<index<<endl;
	//		}
	//	C_YELLOW;
	//	//cout << endl << "���" << n << "��" << busNames[bustype] << endl;
	//	C_NONE;
	//}
	//����Ϊ�ҳ����ڳ��ߵļ��
	cout << "Ѱ�ҳ��ߡ���";
	for (int index = 0; index < objTypeVessel.size(); index++) {
		if (objTypeVessel[index] != 4) continue;
		vector<int>c_what = GetConnect(adjGraphPure, index);
		for (int i = 0; i < c_what.size(); i++) {
			if (objTypeVessel[c_what[i]] != 2)continue;
			int gapno = gapNoVessel[c_what[i]];
			if (gapno == -1)continue;
			for (int j = 0; j < gapNoVessel.size(); j++) {
				if (gapNoVessel[j] != gapno)continue;
				gapTypeVessel[j] += 2;
			}
		}
	}
	cout << "���" << endl;
	//����Ϊ�ж��Ƿ���ڵ�ĸ��/˫ĸ�߷ֶ�
	cout << "ĸ�߷ֶμ�⡭��";
	for (int n = 0; n < gapCount; n++) {
		bool is_ngap_eligible = true;
		for (int index = 0; index < gapNoVessel.size(); index++) {
			if (gapNoVessel[index] != n)continue;
			if (objTypeVessel[index] != 2)continue;
			//����index�����ض������ļ���е�Ԫ��
			if (gapTypeVessel[index] == GAP_NONE) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_ACROSS_OUT) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_NORMAL_OUT) { is_ngap_eligible = false; break; }
			//������ǰѭ����n������һ��
			vector<int> tmp = GetConnect(adjGraphLine, index);
			bool have_a_bus = false;
			for (int i = 0; i < tmp.size(); i++)
				if (busTypeVessel[tmp[i]] != 0) have_a_bus = true;
			is_ngap_eligible = is_ngap_eligible && have_a_bus;
			//if (is_ngap_eligible) cout << "n:" << n << ",index:" << index << endl;
		}
		if (!is_ngap_eligible) continue;
		//cout << n << "�ż��Ϊ" << endl;
		for (int index = 0; index < gapNoVessel.size(); index++) {
			if (gapNoVessel[index] != n)continue;
			if (objTypeVessel[index] != 2)continue;
			//����index�����ض������ļ���е�Ԫ��
			if (gapTypeVessel[index] == GAP_NONE) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_ACROSS_OUT) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_NORMAL_OUT) { is_ngap_eligible = false; break; }
			//������ǰѭ����n������һ��
			vector<int> tmp = GetConnect(adjGraphLine, index);
			for (int i = 0; i < tmp.size(); i++)
				if (busTypeVessel[tmp[i]] != 0) {
					vector<int> allLine;
					switch (busTypeVessel[tmp[i]]) {
					case BUS_DOUBLE://Ϊ˫ĸ��
						switch (gapTypeVessel[index]) {
						default:
						case GAP_NORMAL:
							break;
						case GAP_DBUS_SUBSECTION:
							allLine = GetAllLine(adjGraphLine, tmp[i]);
							for (int j = 0; j < allLine.size(); j++)
								busTypeVessel[allLine[j]] = BUS_DOUBLE_SUB;
							break;
						}
						break;
					default:
					case BUS_://Ϊ��ĸ��
					case BUS_SINGLE:
						switch (gapTypeVessel[index]) {
						default:
						case GAP_NORMAL:
							allLine = GetAllLine(adjGraphLine, tmp[i]);
							for (int j = 0; j < allLine.size(); j++)
								busTypeVessel[allLine[j]] = BUS_SINGLE_SUB;
							for (int k = 0; k < gapNoVessel.size(); k++)
								if (gapNoVessel[k] == n)
									gapTypeVessel[k] = GAP_SBUS_SUBSECTION;
							break;
						}
						break;
					}
				}
		}
	}
	cout << "���" << endl;
	//ĸ�߾ۺ�
	cout << "ĸ�߾ۺ��С���";
	for (int n = 0; n < gapCount; n++) {
		set<int> c_what_bus;
		for (int index = 0; index < gapNoVessel.size(); index++) {
			if (gapNoVessel[index] != n)continue;
			if (!(gapTypeVessel[index] == GAP_NORMAL ||
				gapTypeVessel[index] == GAP_DBUS_SUBSECTION ||
				gapTypeVessel[index] == GAP_SBUS_SUBSECTION)) continue;
			if (objTypeVessel[index] != 2) continue;
			//cout << index << endl;
			vector<int> c_what_line = GetConnect(adjGraphLine, index);
				for (int i = 0; i < c_what_line.size(); i++) {
					if (busTypeVessel[c_what_line[i]] == 0)continue;
					c_what_bus.insert(c_what_line[i]);
				}
		}
		if (c_what_bus.size() == 0)continue;
		int bus_first_no = busNoVessel[*(c_what_bus.begin())];
		for (itr_setint itr = c_what_bus.begin(); itr != c_what_bus.end(); itr++) {
			vector<int> allLine = GetAllLine(adjGraphLine, *itr);
			for (int j = 0; j < allLine.size(); j++) {
				busNoVessel[allLine[j]] = bus_first_no;
			}
		}
	}
	cout << "���" << endl;
	C_YELLOW;
	cout << "ĸ�߼������" << endl;
	C_NONE;
	set<int> s(busNoVessel.begin(), busNoVessel.end());
	s.erase(-1);
	if (s.size() == 0)cout << "��ĸ��" << endl;
	else for (itr_setint itr = s.begin(); itr != s.end(); itr++) {
		int type = 0;
		int out_count = 0;
		for (int index = 0; index < busNoVessel.size(); index++) {
			if (*itr != busNoVessel[index])continue;
			cout << index << "��,";
				type = busTypeVessel[index];
		}
		for (int index = 0; index < objTypeVessel.size(); index++) {
			if (objTypeVessel[index] != 4) continue;
			if (*itr == busNoVessel[NearestBus(adjGraphLine, index)]) out_count++;
		}
		cout <<"\b[����]" << endl << "����" << busNames[type] << ",��ĸ�߳���" << out_count << "��" << endl;
	}

}
void TopoDetectYolo::DetectGapBus2() {
	//�����Ǽ����ⲿ��

	for (int index = 0; index < objTypeVessel.size(); index++) {
		if (objTypeVessel[index] == 3) {//�ȶ�λ�����ж�·��
			vector<int> c_what = GetConnect(adjGraphPure, index);
			if (c_what.size() <= 0) continue;
			vector<int> c_type_count(classes + 2);
			for (int i = 0; i < c_what.size(); i++)
				c_type_count[objTypeVessel[c_what[i]]]++;//������ö�·����ϵ������Ԫ����
			int c_isw_count = c_type_count[2];
			Obj_gap tmp;
			if (c_isw_count <= 1) continue;
			else if (c_isw_count == 2) tmp.type = GAP_NORMAL;
			else if (c_isw_count == 3) tmp.type = GAP_ACROSS;
			else if (c_isw_count == 4) tmp.type = GAP_DBUS_SUBSECTION;
			tmp.Insert(index);
			for (int i = 0; i < c_what.size(); i++)
				if (objTypeVessel[c_what[i]] == 2)
					tmp.Insert(c_what[i]);
			gap_s.push_back(tmp);
		}
	}
	for (int index = 0; index < gap_s.size(); index++) {
		cout << index << "�ż����";
		cout << gapNames[gap_s[index].type] << endl;
		gap_s[index].ShowObjs();
	}
	//return;
	//������ĸ�߳�����ⲿ��
	int busCount = 0;
	for (int index = 0; index < busTypeVessel.size(); index++) {
		if (ObjType(index) != classes + 1)continue;
		if (busTypeVessel[index] == 1)continue;
		vector<int> allLine = GetAllLine(this->adjGraphLine, index);
		//cout << "����" << index << "�źͣ�";
		vector<int> c_type_count(classes + 2);
		for (int i = 0; i < allLine.size(); i++) {
			//cout << allLine[i] << ",";
			vector<int> c_what = GetConnect(adjGraphLine, allLine[i]);
			for (int j = 0; j < c_what.size(); j++)
				c_type_count[objTypeVessel[c_what[j]]]++;
		}
		//cout << endl;
		bool not_a_bus = false;
		if (c_type_count[2] < 2) not_a_bus = true;
		for (int type = 0; type < classes + 1; type++) {
			if (type == 2)continue;
			if (c_type_count[type] > 0) not_a_bus = true;
		}
		if (not_a_bus) continue;
		Obj_bus tmp;
		for (int i = 0; i < allLine.size(); i++) 
			tmp.Insert(allLine[i]);
		tmp.type = 1;
		bus_s.push_back(tmp);

	}
	for (int index = 0; index < bus_s.size(); index++) {
		cout << index << "��ĸ�ߣ�";
		cout << busNames[bus_s[index].type] << endl;
		bus_s[index].ShowObjs();
	}
	//����Ϊ�ҳ����ڳ��ߵļ��
	// 
	//������ĸ�߾��������ⲿ��
	
	

	
	//����Ϊ�ж��Ƿ���ڵ�ĸ��/˫ĸ�߷ֶ�
	

}
void TopoDetectYolo::ShowTopo(int showType) {
	vector<int> connect_what;
	switch (showType) {
	case TOPO_ALL:
		for (int index = 0; index < adjGraph.rows; index++) {
			C_YELLOW;
			cout << index << "��[" << subEleNames[objTypeVessel[index]] << "]������������" << endl;
			C_NONE;
			connect_what = GetConnect(adjGraph, index);
			if (connect_what.size() > 0)
				for (int i = 0; i < connect_what.size(); i++)
					cout << connect_what[i] << "��[" << subEleNames[objTypeVessel[connect_what[i]]] << "],";
			else cout << "��";
			cout << endl;
		}
		break;
	case TOPO_LINE_EXCEPT:
		for (int index = 0; index < adjGraphPure.rows; index++) {
			if (objTypeVessel[index] == classes + 1) continue;
			if (objTypeVessel[index] == classes) continue;
			if (objTypeVessel[index] == classes - 1) continue;
			C_YELLOW;
			cout << index << "��[" << subEleNames[objTypeVessel[index]] << "]������������" << endl;
			C_NONE;
			//for (int i = 0; i < adjGraphPure.cols; i++)
			//	if (adjGraphPure.at<float>(index, i) != 0)
			//		if (index != i)
			//			cout << i << "��[" << subEleNames[objTypeVessel[i]] << "],";
			connect_what = GetConnect(adjGraphPure, index);
			if (connect_what.size() > 0)
				for (int i = 0; i < connect_what.size(); i++)
					cout << connect_what[i] << "��[" << subEleNames[objTypeVessel[connect_what[i]]] << "],";
			else cout << "��";
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
	putText(canva, text, Point(box.x, box.y-1), FONT_HERSHEY_COMPLEX_SMALL, 0.3, Scalar(255, 255, 255), 1);

}
Mat TopoDetectYolo::EliminateType(Mat inGraph,int type) {//��������ͼ�е��ض��������������������������������˴�����
	Mat outGraph = inGraph.clone();
	for (int index = 0; index < objPixelVessel.size(); index++) {
		if (objTypeVessel[index] == type) {
			vector<int>connectElement;//���ڼ�¼�͸��������������ͼԪ�ĺ���
			for (int i = 0; i < objPixelVessel.size(); i++)
				if (outGraph.at<float>(index, i) != 0)
					if (index != i) {
						connectElement.push_back(i);
						outGraph.at<float>(index, i) = 0;
						outGraph.at<float>(i, index) = 0;
					}
			//cout << "����ͼ�е�" << index << "��" << subEleNames[m_type] << endl;
			if (connectElement.size() > 1)
				//��Ϊ0����˵����������������κ�ͼԪ����
				//��Ϊ1����˵�����������ֻ��һ��ͼԪ����
				//������������������޸�ͼԪ�����ӹ�ϵ
				for (int i = 0; i < connectElement.size() - 1; i++)
					for (int j = i + 1; j < connectElement.size(); j++) {
						outGraph.at<float>(connectElement[i], connectElement[j]) = 1;
						outGraph.at<float>(connectElement[j], connectElement[i]) = 1;
						//cout << "�ѽ�" << i << "�ţ�" << j << "������" << endl;
					}
		}
	}
	return outGraph;
}
vector<int> TopoDetectYolo::GetAllLine(Mat adj,int index) {
	vector<int> out{ index };
	if (ObjType(index) != classes + 1) return out;
	Mat adjL = adj.clone();
	vector<int> c_what = GetConnect(adjL, index);
	for (int i = 0; i < c_what.size(); i++) {
		if (ObjType(c_what[i]) != classes + 1)continue;
		adjL.at<float>(c_what[i], index) = 0;
		adjL.at<float>(index, c_what[i]) = 0;
		vector<int> tmp = GetAllLine(adjL, c_what[i]);
		for (int j = 0; j < tmp.size(); j++) out.push_back(tmp[j]);
	}
	return out;
}
int TopoDetectYolo::NearestBus(Mat adj, int index) {
	if (busTypeVessel[index] != 0)return index;
	if (objTypeVessel[index] == 0) return index;
	Mat adjL = adj.clone();
	vector<int> c_what = GetConnect(adjL, index);
	if (c_what.size() == 0) return index;
	for (int i = 0; i < c_what.size(); i++) {
		//cout <<objTypeVessel[c_what[i]] << c_what[i] << endl;
		adjL.at<float>(c_what[i], index) = 0;
		adjL.at<float>(index, c_what[i]) = 0;
		return NearestBus(adjL, c_what[i]);
	}
}
vector<int> GetConnect(Mat adjMat, int  index) {
	vector<int> out;
	if (index >= adjMat.rows || index < 0) {
		cout << "�����±�Խ�磡" << endl;
		//out.push_back(0);
		return out;
	}
	for (int i = 0; i < adjMat.cols; i++) {
		int thisone = adjMat.at<float>(index, i);
		//cout << thisone << ",";
		//cout << endl;
		if (thisone == 1)
			if (index != i) {
				out.push_back(i);
				//cout << i << ",";
			}


	}
	//cout << endl;
	return out;
}
vector<int> GetAllLine(Mat adjMatLine, int index) {
}
