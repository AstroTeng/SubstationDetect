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
	adjGraphLine = adjGraph.clone();
	for (int m_type = classes + 1; m_type >= classes - 1; m_type--)
		adjGraphPure = EliminateType(adjGraphPure, m_type);
	for (int m_type = classes; m_type >= classes - 1; m_type--)
		adjGraphLine = EliminateType(adjGraphLine, m_type);
	
	cout << "拓扑检测完成！" << endl;
}
void TopoDetectYolo::DetectGapBus() {
	//以下是间隔检测部分
	cout << "进行间隔检测……";
	int gapCount = 0;
	for (int index = 0; index < objTypeVessel.size(); index++) {
		if (objTypeVessel[index] == 3) {//先定位到所有断路器
			vector<int> connect_what = GetConnect(adjGraphPure, index);
			if (connect_what.size() > 0) {
				vector<int> connectTypeCount(classes + 2);
				for (int i = 0; i < connect_what.size(); i++)
					connectTypeCount[objTypeVessel[connect_what[i]]]++;//计算与该断路器联系的所有元件的
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
	cout << "完成" << endl;
	//for (int n = 0; n < gapCount; n++) {//间隔显示
	//	int g_type = 0;
	//	for (int index = 0; index < gapNoVessel.size(); index++) 
	//		if (gapNoVessel[index] == n) {
	//			cout << index << "号[" << subEleNames[objTypeVessel[index]] << "],";
	//			g_type = gapTypeVessel[index];
	//		}
	//	cout << endl << "组成" << n << "号" << gapNames[g_type] << endl;
	//}
	//以下是母线初步检测部分
	cout << "进行母线初步检测……";
	int busCount = 0;
	for (int index = 0; index < busTypeVessel.size(); index++) {
		if (ObjType(index) != classes + 1)continue;
		if (busTypeVessel[index] == 1)continue;
		vector<int> allLine = GetAllLine(this->adjGraphLine, index);
		//cout << "导线" << index << "号和：";
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
	cout << "完成" << endl;
	//以下是母线具体种类检测部分
	cout << "进行母线分类……" ;
	for (int n = 0; n < busCount; n++) {
		vector<int> g_type_count(3);//与当前n号母线相连的间隔种类计数
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
	cout << "完成" << endl;
	//for (int n = 0; n < busCount; n++) {
	//	int bustype = 0;
	//	for (int index = 0; index < busNoVessel.size(); index++)
	//		if (n == busNoVessel[index]) {
	//			//cout << index << "号导线，";
	//			bustype = busTypeVessel[index];
	//			//cout << "bustype" << bustype << "index"<<index<<endl;
	//		}
	//	C_YELLOW;
	//	//cout << endl << "组成" << n << "号" << busNames[bustype] << endl;
	//	C_NONE;
	//}
	//以下为找出属于出线的间隔
	cout << "寻找出线……";
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
	cout << "完成" << endl;
	//以下为判断是否存在单母线/双母线分段
	cout << "母线分段检测……";
	for (int n = 0; n < gapCount; n++) {
		bool is_ngap_eligible = true;
		for (int index = 0; index < gapNoVessel.size(); index++) {
			if (gapNoVessel[index] != n)continue;
			if (objTypeVessel[index] != 2)continue;
			//至此index代表特定号数的间隔中的元素
			if (gapTypeVessel[index] == GAP_NONE) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_ACROSS_OUT) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_NORMAL_OUT) { is_ngap_eligible = false; break; }
			//跳出当前循环，n换到下一个
			vector<int> tmp = GetConnect(adjGraphLine, index);
			bool have_a_bus = false;
			for (int i = 0; i < tmp.size(); i++)
				if (busTypeVessel[tmp[i]] != 0) have_a_bus = true;
			is_ngap_eligible = is_ngap_eligible && have_a_bus;
			//if (is_ngap_eligible) cout << "n:" << n << ",index:" << index << endl;
		}
		if (!is_ngap_eligible) continue;
		//cout << n << "号间隔为" << endl;
		for (int index = 0; index < gapNoVessel.size(); index++) {
			if (gapNoVessel[index] != n)continue;
			if (objTypeVessel[index] != 2)continue;
			//至此index代表特定号数的间隔中的元素
			if (gapTypeVessel[index] == GAP_NONE) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_ACROSS_OUT) { is_ngap_eligible = false; break; }
			if (gapTypeVessel[index] == GAP_NORMAL_OUT) { is_ngap_eligible = false; break; }
			//跳出当前循环，n换到下一个
			vector<int> tmp = GetConnect(adjGraphLine, index);
			for (int i = 0; i < tmp.size(); i++)
				if (busTypeVessel[tmp[i]] != 0) {
					vector<int> allLine;
					switch (busTypeVessel[tmp[i]]) {
					case BUS_DOUBLE://为双母线
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
					case BUS_://为单母线
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
	cout << "完成" << endl;
	//母线聚合
	cout << "母线聚合中……";
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
	cout << "完成" << endl;
	C_YELLOW;
	cout << "母线检测结果：" << endl;
	C_NONE;
	set<int> s(busNoVessel.begin(), busNoVessel.end());
	s.erase(-1);
	if (s.size() == 0)cout << "无母线" << endl;
	else for (itr_setint itr = s.begin(); itr != s.end(); itr++) {
		int type = 0;
		int out_count = 0;
		for (int index = 0; index < busNoVessel.size(); index++) {
			if (*itr != busNoVessel[index])continue;
			cout << index << "号,";
				type = busTypeVessel[index];
		}
		for (int index = 0; index < objTypeVessel.size(); index++) {
			if (objTypeVessel[index] != 4) continue;
			if (*itr == busNoVessel[NearestBus(adjGraphLine, index)]) out_count++;
		}
		cout <<"\b[导线]" << endl << "构成" << busNames[type] << ",该母线出线" << out_count << "回" << endl;
	}

}
void TopoDetectYolo::DetectGapBus2() {
	//以下是间隔检测部分

	for (int index = 0; index < objTypeVessel.size(); index++) {
		if (objTypeVessel[index] == 3) {//先定位到所有断路器
			vector<int> c_what = GetConnect(adjGraphPure, index);
			if (c_what.size() <= 0) continue;
			vector<int> c_type_count(classes + 2);
			for (int i = 0; i < c_what.size(); i++)
				c_type_count[objTypeVessel[c_what[i]]]++;//计算与该断路器联系的所有元件的
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
		cout << index << "号间隔：";
		cout << gapNames[gap_s[index].type] << endl;
		gap_s[index].ShowObjs();
	}
	//return;
	//以下是母线初步检测部分
	int busCount = 0;
	for (int index = 0; index < busTypeVessel.size(); index++) {
		if (ObjType(index) != classes + 1)continue;
		if (busTypeVessel[index] == 1)continue;
		vector<int> allLine = GetAllLine(this->adjGraphLine, index);
		//cout << "导线" << index << "号和：";
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
		cout << index << "号母线：";
		cout << busNames[bus_s[index].type] << endl;
		bus_s[index].ShowObjs();
	}
	//以下为找出属于出线的间隔
	// 
	//以下是母线具体种类检测部分
	
	

	
	//以下为判断是否存在单母线/双母线分段
	

}
void TopoDetectYolo::ShowTopo(int showType) {
	vector<int> connect_what;
	switch (showType) {
	case TOPO_ALL:
		for (int index = 0; index < adjGraph.rows; index++) {
			C_YELLOW;
			cout << index << "号[" << subEleNames[objTypeVessel[index]] << "]和下述相连：" << endl;
			C_NONE;
			connect_what = GetConnect(adjGraph, index);
			if (connect_what.size() > 0)
				for (int i = 0; i < connect_what.size(); i++)
					cout << connect_what[i] << "号[" << subEleNames[objTypeVessel[connect_what[i]]] << "],";
			else cout << "无";
			cout << endl;
		}
		break;
	case TOPO_LINE_EXCEPT:
		for (int index = 0; index < adjGraphPure.rows; index++) {
			if (objTypeVessel[index] == classes + 1) continue;
			if (objTypeVessel[index] == classes) continue;
			if (objTypeVessel[index] == classes - 1) continue;
			C_YELLOW;
			cout << index << "号[" << subEleNames[objTypeVessel[index]] << "]和下述相连：" << endl;
			C_NONE;
			//for (int i = 0; i < adjGraphPure.cols; i++)
			//	if (adjGraphPure.at<float>(index, i) != 0)
			//		if (index != i)
			//			cout << i << "号[" << subEleNames[objTypeVessel[i]] << "],";
			connect_what = GetConnect(adjGraphPure, index);
			if (connect_what.size() > 0)
				for (int i = 0; i < connect_what.size(); i++)
					cout << connect_what[i] << "号[" << subEleNames[objTypeVessel[connect_what[i]]] << "],";
			else cout << "无";
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
	putText(canva, text, Point(box.x, box.y-1), FONT_HERSHEY_COMPLEX_SMALL, 0.3, Scalar(255, 255, 255), 1);

}
Mat TopoDetectYolo::EliminateType(Mat inGraph,int type) {//消除拓扑图中的特定物件，并将与其相连的所有其它物件彼此相连
	Mat outGraph = inGraph.clone();
	for (int index = 0; index < objPixelVessel.size(); index++) {
		if (objTypeVessel[index] == type) {
			vector<int>connectElement;//用于记录和该类型物件相连的图元的号码
			for (int i = 0; i < objPixelVessel.size(); i++)
				if (outGraph.at<float>(index, i) != 0)
					if (index != i) {
						connectElement.push_back(i);
						outGraph.at<float>(index, i) = 0;
						outGraph.at<float>(i, index) = 0;
					}
			//cout << "对于图中的" << index << "号" << subEleNames[m_type] << endl;
			if (connectElement.size() > 1)
				//若为0，则说明本连接物件不与任何图元相连
				//若为1，则说明本连接物件只和一个图元相连
				//以上两种情况都不用修改图元的连接关系
				for (int i = 0; i < connectElement.size() - 1; i++)
					for (int j = i + 1; j < connectElement.size(); j++) {
						outGraph.at<float>(connectElement[i], connectElement[j]) = 1;
						outGraph.at<float>(connectElement[j], connectElement[i]) = 1;
						//cout << "已将" << i << "号，" << j << "号连接" << endl;
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
		cout << "错误：下标越界！" << endl;
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
