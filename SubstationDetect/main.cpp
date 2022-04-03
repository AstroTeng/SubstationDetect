#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
#include "ElementGenerate.h"
#include "CNNet.h"
#include "NeuronNet.h"
#include "CoutShape.h"
#include "Detect.h"
#include "TopoDetect.h"
#include "TopoDetectYolo.h"
#include <yolo_v2_class.hpp>
using namespace std;
using namespace cv;
static String nameBuffer1[8] = {
	"3pt",//三相变压器
	"2pt",//变压器
	"isw",//隔离开关
	"brk",//断路器
	"ld",//负载
	"gnd",//接地
	"crs",//十字交叉
	"no"//无
};
static String nameBuffer2[TOTAL_ELEMENT] = {
	"3pt",//三相变压器
	"2pt",//变压器
	"isw",//隔离开关
	"brk",//断路器
	"ld",//负载
	"gnd",//接地
};
static String nameBuffer3[2] = {
	"无",
	"有"
};

//https://zhuanlan.zhihu.com/p/115571464 深度学习 | 反向传播详解
// https://zhuanlan.zhihu.com/p/61863634 全连接神经网络中反向传播算法数学推导
//https://zhuanlan.zhihu.com/p/61898234 卷积神经网络(CNN)反向传播算法推导
//https://zhuanlan.zhihu.com/p/62303214 卷积神经网络(CNN)的底层实现——以LeNet为例

#define clear() cout << "\033c" << flush
#define START_WITH_RANDOM 1
#define START_WITH_DATA 0
#define LEARNING_RATE 0.01
#define BATCH 5
#define EPOCH 1
#define WEIGHT_PATH "data-new\\"
#define IMG_PATH "pics\\generated-elements\\"
#define RESULT_PATH "pics\\test-new\\"
#define WEIGHT_SUFFIX "dat"


vector<Rect> CalcuCentroid(vector<Rect> oriRect, Mat img) {
	vector<Rect> out;
	Mat cpy2 = img.clone();
	threshold(cpy2, cpy2, 180, 255, THRESH_BINARY_INV);
	cvtColor(cpy2, cpy2, COLOR_RGB2GRAY);
	cout << "正在计算各线框质心……";
	for (int i = 0; i < oriRect.size() ; i++) {//调整中心点
		double X = 0, Y = 0, T = 0;
		for (int m = oriRect[i].y; m < oriRect[i].y + oriRect[i].height; m++)
			for (int n = oriRect[i].x; n < oriRect[i].x + oriRect[i].width; n++) {
				T += (double)cpy2.at<uchar>(m, n);
				Y += (double)cpy2.at<uchar>(m, n) * m;
				X += (double)cpy2.at<uchar>(m, n) * n;
			}
		//cout << i << " " << oriRect.size() << " " << endl;
		Y /= T;
		X /= T;
		Point core(X, Y);
		//rectangle(cpy, oriRect[i], Scalar(200,0,0), 1);
		//circle(cpy, core, 1, Scalar(0, 0, 200),1);
		Rect fined = oriRect[i];
		double finedLength = max(fined.width, fined.height);
		fined.x = core.x - finedLength / 2;
		fined.y = core.y - finedLength / 2;
		fined.width = finedLength;
		fined.height = finedLength;
		out.push_back(fined);
	}
	cout << "完成" << endl;
	return out;
}
void GeneratePipeline() {//此函数中存放图元生成流程5
	RNG rng(getTickCount());
	vector<String>pathVessel;
	PathFetch(pathVessel, "jpg", IMG_PATH);
	for (int i = 0; i < pathVessel.size(); i++) {
		char buffer[256];
		strcpy_s(buffer, pathVessel[i].c_str());
		remove(buffer);
	}
	std::default_random_engine random(time(NULL));
	std::uniform_int_distribution<int> dis1(0, TOTAL_ELEMENT-1);
	ElementGenerate T(128, 128, IMG_PATH);
	for (int i = 0; i < 1000; i++) {
		switch (dis1(random)) {//在[0,TOTAL_ELEMENT+1)上均匀生成
		case 0:
			T.Transformer3P(rng);
			break;
		case 1:
			T.Transformer(rng);
			break;
		case 2:
			T.IsoSwitch(rng);
			break;
		case 3:
			T.Breaker(rng);
			break;
		case 4:
			T.Load(rng);
			break;
		case 5:
			T.Ground(rng);
			break;
		}
		//T.Breaker(rng);
		rng.next();
	}
}
void DetectPipeline(String d_path,Net *d_net,String dat_path) {

#define DES_HEIGHT 3000
#define FIN_HEIGHT 750
	Mat ori = imread(d_path);
	Mat show;
	Mat cpy = ori.clone();
	TopoDetect substationTopo;
	double ratio = DES_HEIGHT / (double)ori.rows;
	double ratio2 = FIN_HEIGHT / (double)cpy.rows;
	resize(ori, ori, Size(0, 0), ratio, ratio);
	resize(cpy, cpy, Size(0, 0), ratio2, ratio2);
	substationTopo.SetSize(cpy.size());
	Mat array3 = (Mat_<double>(3, 3) <<
		1, -1, 1,
		-1, 1, -1,
		1, -1, 1);
	filter2D(ori, ori, -1, array3, Point(-1, -1), 0, BORDER_CONSTANT);
	threshold(ori, ori, 200, 255, THRESH_BINARY);
	cvtColor(ori, ori, COLOR_RGB2GRAY);
	Canny(ori, ori, 3, 9, 3);
	imwrite("1.jpg", ori);

	Mat array1 = (Mat_<double>(3, 3) <<
		-1, -1, -1,
		0, 0, 0,
		1, 1, 1);
	Mat array2 = (Mat_<double>(3, 3) <<
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	cout << "正在预处理……" ;
	filter2D(ori, ori, -1, array1, Point(-1, -1), 0, BORDER_CONSTANT);
	filter2D(ori, ori, -1, array2, Point(-1, -1), 0, BORDER_CONSTANT);
	findContours(ori, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) <= 2)
			fillPoly(ori, contours[i], Scalar(0, 0, 0));
	}
	cout << "完成" << endl;
	//imwrite("pics\\test-new\\去除横竖线.jpg", ori);
	//resize(ori, show, cpy.size());
	//imshow("完成直线清除", show);
	//waitKey();

	cout << "正在膨胀……";
	Mat e1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	for (int i = 0; i < 10; i++) {
		dilate(ori, ori, e1, Point(-1, -1), 1, BORDER_CONSTANT);
	}
	cout << "完成" << endl;
	//resize(ori, show, cpy.size());
	//imshow("完成膨胀", show);
	//waitKey();
	cout << "正在腐蚀……";
	Mat e2 = getStructuringElement(MORPH_RECT, Size(5, 5));
	for (int i = 0; i < 1; i++) {
		erode(ori, ori, e2, Point(-1, -1), 1, BORDER_CONSTANT);
	}
	cout << "完成" << endl;
	//resize(ori, show, cpy.size());
	//imshow("完成腐蚀", show);
	//waitKey();
	//dilate(ori, ori, e2, Point(-1, -1), 1, BORDER_CONSTANT);
	//erode(ori, ori, e2, Point(-1, -1), 1, BORDER_CONSTANT);
	resize(ori, ori, cpy.size());
	threshold(ori, ori, 0, 255, THRESH_BINARY);

	cout << "正在合并零散线框……";
	findContours(ori, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	Mat canv(cpy.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < contours.size(); i++) {
		Rect tmp = boundingRect(contours[i]);
		rectangle(canv, tmp, Scalar(255), -1);
	}
	findContours(canv, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cout << "完成" << endl;
#define EXTEND_PIXEL 2
	vector<Rect> resultRect;
	vector<Rect> finedRect;
	vector<Rect> lineRect;
	for (int i = 0; i < contours.size(); i++) {
		Rect tmp = boundingRect(contours[i]);
		tmp.x -= EXTEND_PIXEL;
		tmp.width += EXTEND_PIXEL * 2;
		tmp.y -= EXTEND_PIXEL;
		tmp.height += EXTEND_PIXEL * 2;
		resultRect.push_back(tmp);
		//rectangle(cpy, tmp, Scalar(200), 1);
	}
	Mat cpy2 = cpy.clone();
	threshold(cpy2, cpy2, 180, 255, THRESH_BINARY_INV);
	cvtColor(cpy2, cpy2, COLOR_RGB2GRAY);
	cout << "正在计算各线框质心……";
	for (int i = 0; i < resultRect.size() - 1; i++) {//调整中心点
		double X = 0, Y = 0, T = 0;
		for (int m = resultRect[i].y; m < resultRect[i].y + resultRect[i].height; m++)
			for (int n = resultRect[i].x; n < resultRect[i].x + resultRect[i].width; n++) {
				T += (double)cpy2.at<uchar>(m, n);
				Y += (double)cpy2.at<uchar>(m, n) * m;
				X += (double)cpy2.at<uchar>(m, n) * n;
			}
		//cout << i << " " << resultRect.size() << " " << endl;
		Y /= T;
		X /= T;
		Point core(X, Y);
		//rectangle(cpy, resultRect[i], Scalar(200,0,0), 1);
		//circle(cpy, core, 1, Scalar(0, 0, 200),1);
		Rect fined = resultRect[i];
		double finedLength = max(fined.width, fined.height);
		fined.x = core.x - finedLength / 2;
		fined.y = core.y - finedLength / 2;
		fined.width = finedLength;
		fined.height = finedLength;
		finedRect.push_back(fined);
	}
	cout << "完成" << endl;


	(*d_net).LoadData(dat_path);
	ori = cpy.clone();
	Mat ori2 = cpy.clone();
	threshold(ori, ori, 190, 255, THRESH_BINARY);
	if (0)//此处设为1则进入标记模式,否则为普通检测模式
		for (int i = 0; i < finedRect.size(); i++) {//给图片打标签用
			string savepath = "pics\\seg-train\\";
			string no = to_string(13000 + i);
			no[0] = 'N';
			savepath += no;
			rectangle(cpy, finedRect[i], Scalar(0, 0, 200));
			imshow("1", cpy);

			int lable;
			lable = waitKey();
			lable -= '0';
			destroyWindow("1");
			savepath += "_" + to_string(lable) + "_.jpg";
			imwrite(savepath, ori(finedRect[i]));
			rectangle(cpy, finedRect[i], Scalar(0, 200, 0));
			//return;
		}
	for (int i = 0; i < finedRect.size(); i++) {//绘制矩形框
		imwrite("tmp.jpg", ori(finedRect[i]));
		(*d_net).Target("tmp.jpg", false);
		(*d_net).Forward();
		int max_lable = (*d_net).ShowResult();
		substationTopo.AddObj(finedRect[i], max_lable);//在此处往拓扑识别类中添加图元矩形框
		if (max_lable != 7) {
			rectangle(cpy, finedRect[i], Scalar(0, 252, 124));
			putText(cpy, (*d_net).n_buff[max_lable],
				Point(finedRect[i].x, finedRect[i].y + 10),
				FONT_HERSHEY_DUPLEX, 0.3, Scalar(133, 21, 199), 1);
			rectangle(ori2, finedRect[i], Scalar(255, 255, 255),-1);
		}
		else {
			rectangle(cpy, finedRect[i], Scalar(150, 150, 150));
		}
		imshow("1", cpy);
		waitKey(1);
	}
	//waitKey();
	//destroyWindow("1");

	threshold(ori2, ori2, 200, 255, THRESH_BINARY);
	cvtColor(ori2, ori2, COLOR_RGB2GRAY);
	Canny(ori2, ori2, 3, 9, 3);
	Mat e3 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	for (int i = 0; i < 1; i++) {
		dilate(ori2, ori2, e3, Point(-1, -1), 1, BORDER_CONSTANT);
	}
	vector<vector<Point>> lineCon;
	findContours(ori2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	lineCon = contours;
	for (int i = 0; i < contours.size(); i++) {//绘制导线
		substationTopo.AddObj(contours[i]);//在此处往拓扑识别类中添加导线自由线框
		//drawContours(cpy, contours, i, Scalar(255), 1);
		//imshow("1", cpy);
		//waitKey(1);
	}
	for (int i = 0; i < substationTopo.objPixelVessel.size(); i++) {
		int typenow = substationTopo.objTypeVessel[i];
		if(typenow!=6&&typenow!=7&&typenow!=8)
			putText(cpy, to_string(i), substationTopo.objPixelVessel[i][0],
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 1, 0), 1);
	}
		

	//substationTopo.ShowInPic();
	substationTopo.ResetGraph();
    substationTopo.DetectTopo();
	substationTopo.ShowTopo(TOPO_LINE_EXCEPT);
	//imshow("adj",Mat2Pic(substationTopo.graph.adjMat));
	//waitKey();

	imshow("1", cpy);
	waitKey();
	destroyWindow("1");
	return;

}
void DetectPipeline2(string cfg_path, string weight_path, string img_path) {
	TopoDetectYolo subTopo(9);
	Detector subDetect(cfg_path, weight_path);
	vector<Rect> eleRects;

	Mat img = imread(img_path);
	Mat img2 = img.clone();


	double ratio = 750 / (double)img.rows;
	//resize(img, img, Size(0, 0), ratio, ratio);
	vector<bbox_t> eleBoxes = subDetect.detect(img_path);

	for (int index = 0; index < eleBoxes.size(); index++) {
		Rect rect(eleBoxes[index].x, eleBoxes[index].y, eleBoxes[index].w, eleBoxes[index].h);
		eleRects.push_back(rect);
	}
	vector<Rect> eleRectsFined = CalcuCentroid(eleRects, img);
	for (int index = 0; index < eleRectsFined.size(); index++) {
		rectangle(img2, eleRectsFined[index], Scalar(255, 255, 255), -1);
		subTopo.AddObj(eleRectsFined[index], eleBoxes[index].obj_id);
	}
		

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	threshold(img2, img2, 200, 255, THRESH_BINARY);
	cvtColor(img2, img2, COLOR_RGB2GRAY);
	Canny(img2, img2, 3, 9, 3);
	Mat e3 = getStructuringElement(MORPH_CROSS, Size(5, 5));
	for (int i = 0; i < 1; i++) {
		dilate(img2, img2, e3, Point(-1, -1), 1, BORDER_CONSTANT);
	}
	findContours(img2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++) {//绘制导线
		subTopo.AddObj(contours[i]);//在此处往拓扑识别类中添加导线自由线框
		//drawContours(img, contours, i, Scalar(0,0,255), 1);
		//imshow("1", img);
		//waitKey(1);
	}
	for (int index = 0; index < subTopo.objPixelVessel.size(); index++) {
		Scalar color(0, 0, 255);
		if (subTopo.objTypeVessel[index] == subTopo.classes - 1) continue;
		if (subTopo.objTypeVessel[index] == subTopo.classes ) continue;
		if (subTopo.objTypeVessel[index] == subTopo.classes + 1) continue;
		putText(img, to_string(index),
			subTopo.objPixelVessel[index][0], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(200, 20, 200),2);

	}

	subTopo.SetSize(img2.size());
	subTopo.ResetGraph();
	subTopo.DetectTopo();
	subTopo.ShowTopo(TOPO_LINE_EXCEPT);
	imshow("img", img);
	waitKey();
}
int main()
{

	
	//卷积核训练测试代码
	/*Layer_cnn l1(1, 20, 21);
	Layer_cnn l2(20, 1, 21);
	l1.Randomize();
	l2.Randomize();
	Matrix_mat in ;
	Matrix_mat last;
	in = PicLoad2("pics\\2.jpg", false);
	last = PicLoad2("pics\\2-标记.jpg", false);
	resize(in.element[0][0], in.element[0][0], Size(in.element[0][0].cols * (400.0 / in.element[0][0].rows), 400));
	resize(last.element[0][0], last.element[0][0], Size(last.element[0][0].cols * (400.0 / last.element[0][0].rows), 400));
	Matrix_mat tmp;
		l1.Forward(in, ACT_RELU);
		l2.Forward(l1.output_a, ACT_RELU);
		normalize(l2.output_a.element[0][0], l2.output_a.element[0][0], 0, 1, NORM_MINMAX);
	
		tmp = l2.output_a - last;
		l2.Backward(tmp, ACT_RELU);
		PicOutStack(l2.delta_in, "pics\\test-new\\delta");
		l1.Backward(l2.delta_out, ACT_RELU);

		l1.Grad(in);
		l2.Grad(l1.output_a);

		l1.Update(0.001);
		l2.Update(0.001);

		l1.Forward(in, ACT_RELU);
		l2.Forward(l1.output_a, ACT_RELU);
		

		PicOutStack(l1.output_a, "pics\\test-new\\l1");
		PicOutStack(l2.output_a, "pics\\test-new\\l2");
		PicOutStack(l1.conv, "pics\\test-new\\conv1");
		PicOutStack(l2.conv, "pics\\test-new\\conv2");



	return 0;*/

	//生成随机线
	//RNG rng(getTickCount());
	//ElementGenerate Line(128, 128, "pics\\seg-train\\");
	//for (int i = 0; i < 152; i++) {
	//	Line.PureLine(rng);
	//	rng.next();
	//}

	//S.ShowSeg();
	//S.SegUp();
	//LableUp("pics\\segement-elementsA\\");
	//return 0;
	//while (1){
	//	String ipath;
	//	cin >> ipath;
	//	Mat src = imread(ipath);
	//	ImgBinar(src);
	//	imshow(" ", src);
	//	waitKey(0);
	//}
	/*Net Seg1("Seg1", 16, 16, 10);
	Seg1.Add(Layer_cnn(1, 4, 5));
	Seg1.Add(Layer_pool(4, 64));
	Seg1.Add(Layer_cnn(4, 10, 5));
	Seg1.Add(Layer_pool(10, 32));
	Seg1.Add(Layer_fc(Seg1.l_row * Seg1.l_col * Seg1.l_chan, 40));
	Seg1.Add(Layer_fc(40, 2));
	Seg1.n_buff = nameBuffer3;

	Net Seg12("Seg12", 16, 16, 15);
	Seg12.Add(Layer_cnn(1, 5, 3));
	Seg12.Add(Layer_cnn(5, 10, 3));
	Seg12.Add(Layer_pool(10, 64));
	Seg12.Add(Layer_cnn(10, 15, 5));
	Seg12.Add(Layer_pool(15, 32));
	Seg12.Add(Layer_fc(Seg12.l_row * Seg12.l_col * Seg12.l_chan, 40));
	Seg12.Add(Layer_fc(40, 2));
	Seg12.n_buff = nameBuffer3;

	Net New1("New1", 16, 16, 20);
	New1.Add(Layer_cnn(1, 5, 3));
	New1.Add(Layer_cnn(5, 10, 3));
	New1.Add(Layer_pool(10, 64));
	New1.Add(Layer_cnn(10, 20, 5));
	New1.Add(Layer_pool(20, 32));
	New1.Add(Layer_fc(New1.l_row * New1.l_col * New1.l_chan, 120));
	New1.Add(Layer_fc(120, 6));
	New1.n_buff = nameBuffer2;*/
	

	Net New2("New2", 16, 16, 20);
	New2.Add(Layer_cnn(1, 5, 3));
	New2.Add(Layer_cnn(5, 10, 3));
	New2.Add(Layer_pool(10, 64));
	New2.Add(Layer_cnn(10, 20, 5));
	New2.Add(Layer_pool(20, 32));
	New2.Add(Layer_fc(New2.l_row* New2.l_col* New2.l_chan, 160));
	New2.Add(Layer_fc(160, 8));
	New2.n_buff = nameBuffer1;

	//GeneratePipeline();
	if (0) {
		DetectPipeline("pics\\draw1.jpg", &New2, "data-new\\New2-c4.dat");
		return 0;
	}
	else if (1) {
		DetectPipeline2("data\\obj.cfg", "data\\obj_last8000.weights", "pics\\neatopo1.jpg");
		return 0;
	}

	vector<String>imgVessel;
	PathFetch(imgVessel, "jpg", IMG_PATH);

	cout << "已从" << IMG_PATH << "载入" << imgVessel.size() << "张图片" << endl;
	
	//return 0;
	Net net = New2
		;
	ElementGenerate T(128, 128, IMG_PATH);
	while (1) {
		vector<String>pathVessel;
		int op;
		int img_num;
		int fileNo;
		int max_index;
		int error_count;
		char no[32];
		char yn;
		RNG rng(getTickCount());
		String n_name;
		String savePath = WEIGHT_PATH;
		ofstream LOSS;

		default_random_engine random(time(NULL));
		uniform_int_distribution<int> dis1(0, 5);
		int j = 1;
		C_YELLOW;
		cout <<endl<< "当前网络：" <<net.name ;
		C_NONE;
		cout << endl << "请输入操作（0:执行随机化 1:读取权重文件 2:保存权重文件 3:执行预测 ";
		cout << "4:输出结果图片 5:执行训练序列 6:执行测试序列 7:重新生成图元 8:重载图片 9: 退出程序）：" << endl;
		cin >> op;
		switch (op) {
		default://结束程序
		case 9:
			return 0;
			break;
		case 0://执行权重随机化
			net.Randomize();
			break;
		case 1://读取权重文件
			PathFetch(pathVessel, WEIGHT_SUFFIX, WEIGHT_PATH);
			cout << "输入要读取的序号：";
			cin >> no;
			fileNo = atoi(no);
			if (fileNo >= pathVessel.size() || fileNo < 0) { cout << "无效序号！" << endl; break; }
			net.LoadData(pathVessel[fileNo]);
			break;
		case 2://保存权重文件
			cout << "将在" << WEIGHT_PATH << "保存网络的当前权重" << endl;
			cout << "请输入权重文件名：";
			cin >> n_name;
			savePath += n_name + "." + WEIGHT_SUFFIX;
			net.SaveData(savePath);
			break;
		case 3://执行前向传播
			cout << "输入要进行预测的图片路径：" << endl;
			cin >> n_name;
			net.Target(n_name,false);
			net.Forward();
			max_index=net.ShowResult();
			C_CYAN;
			cout << "预测结果：" << net.n_buff[max_index] << endl;
			C_NONE;
			//double loss;
			//loss = -(log(net.seq_fc.back().output_a.element[net.target_lable][0]));//交叉熵作为损失函数
			//cout << "对于标签 " << net.target_lable << " 的损失值为 " << loss << endl;
			break;
		case 4:
			net.OutPic(L_CNN, 0, E_OUTA, RESULT_PATH);
			net.OutPic(L_CNN, 0, E_OUTZ, RESULT_PATH);
			net.OutPic(L_CNN, 1, E_OUTZ, RESULT_PATH);
			net.OutPic(L_CNN, 1, E_OUTA, RESULT_PATH);
			net.OutPic(L_CNN, 2, E_OUTZ, RESULT_PATH);
			net.OutPic(L_CNN, 2, E_OUTA, RESULT_PATH);
			net.OutPic(L_CNN, 2, E_CONV, RESULT_PATH);
			break;
		case 5://训练序列
			LOSS.open("data-new\\LOSS.txt");
			for (int epoch = 0; epoch < EPOCH; epoch++) {
				cout << "第" << epoch << "个序列：" << endl;
				for (int index = 0; index < imgVessel.size(); index++) {
					clear();
					net.Target(imgVessel[index],true);
					C_YELLOW;
					cout << "第" << index << "次训练：" << endl;
					C_NONE;
					cout << "对" << net.target << "进行操作" << endl;
					net.Forward();
					net.ShowEnd();

					double loss;
					loss = -(log(net.seq_fc.back().output_a.element[net.target_lable][0]));//交叉熵作为损失函数
					cout << "对于标签 " << net.target_lable << " 的损失值为 " << loss << endl;
					LOSS << loss << endl;
					net.Backward();
					net.GradUp();
					if (index == j * BATCH - 1) {
						net.Update(LEARNING_RATE / BATCH);
						j++;
					}
				}
			}
			LOSS.close();
			break;
		case 6://测试序列
			error_count = 0;
			for (int index = 0; index < imgVessel.size(); index++) {
				net.Target(imgVessel[index], true);
				net.Forward();
				max_index = net.ShowResult();
				cout << "预测结果：" << net.n_buff[max_index] << endl;
				if (max_index != net.target_lable) error_count++;
				cout << "错误预测数量/测试总数:" << error_count << "/" << index + 1 << endl;
			}
			C_YELLOW;
			cout << "该批测试正确率：" << (1 - error_count * 1.0 / imgVessel.size()) * 100 << "%" << endl;
			C_NONE;
			break;
		case 7:
			cout << "请输入重新生成的图元数量：";
			cin >> no;
			img_num = atoi(no);

			cout << "重新生成前是否清空" << IMG_PATH << "下的文件？(y/n)";
			cin >> yn;
			switch (yn) {
			default:
			case 'n':
				cout << "已选择不清空" << endl;
				break;
			case 'y':
				for (int i = 0; i < imgVessel.size(); i++) {
					char buffer[256];
					strcpy_s(buffer, imgVessel[i].c_str());
					remove(buffer);
				}
				cout << "已清空" << endl;
				break;
			}
			for (int i = 0; i < img_num; i++) {
				switch (dis1(random)) {//在[0,TOTAL_ELEMENT+1)上均匀生成
				case 0:
					T.Transformer3P(rng);
					break;
				case 1:
					T.Transformer(rng);
					break;
				case 2:
					T.IsoSwitch(rng);
					break;
				case 3:
					T.Breaker(rng);
					break;
				case 4:
					T.Load(rng);
					break;
				case 5:
					T.Ground(rng);
					break;
				}
				//T.Breaker(rng);
				rng.next();
			}
			break;
		case 8:
			PathFetch(imgVessel, "jpg", IMG_PATH);
			C_YELLOW;
			cout << "已从" << IMG_PATH << "载入" << imgVessel.size() << "张图片" << endl;
			C_NONE;
			cout << "是否打乱图片顺序？(y/n)";
			cin >> yn;
			switch (yn)
			{
			default:
			case 'n':
				cout << "已选择不打乱" << endl;
				break;
			case 'y':
				srand(unsigned(time(0)));
				random_shuffle(imgVessel.begin(), imgVessel.end());
				for (vector<String>::iterator ptr = imgVessel.begin(); ptr != imgVessel.end(); ptr++)
					cout << ptr - imgVessel.begin() << ". " << *ptr << endl;
				cout << endl;

				cout << "已选择打乱" << endl;
				break;
			}
			cout << "是否在图像上添加随机边框？(y/n)";
			cin >> yn;
			switch (yn) {
			case'n':
				cout << "已选择不添加边框" << endl;
				break;
			case 'y':
				for (int i = 0; i < imgVessel.size(); i++) {
					imwrite(imgVessel[i], RandomPic(imread(imgVessel[i])));
				}
				cout << "已随机添加边框完成" << endl;
				break;
			}
			break;
		}
	}
	return 0;
}
void TrainingPipeline() {
	
}
//void OutputPrint(vector<Mat> output, String path) {//利用该函数将一层的特征图归一化后输出
//	for (int i = 0; i < output.size(); i++) {
//		Mat show;
//		normalize(output[i], show, 0, 255, NORM_MINMAX);
//		imwrite(path + to_string(i) + ".jpg", show);
//	}
//}
//void AllRandomization() {
//	cnn1.Randomization();
//	cnn2.Randomization();
//	cnn3.Randomization();
//	fc1.Randomization();
//	fc2.Randomization();
//}
//void AllLoad() {
//	cnn1.LoadData("data\\1.cnn");
//	cnn2.LoadData("data\\2.cnn");
//	cnn3.LoadData("data\\3.cnn");
//	fc1.LoadData("data\\1.fc");
//	fc2.LoadData("data\\2.fc");
//}
//void AllSave() {
//	cnn1.SaveData("data\\1.cnn");
//	cnn2.SaveData("data\\2.cnn");
//	cnn3.SaveData("data\\3.cnn");
//	fc1.SaveData("data\\1.fc");
//	fc2.SaveData("data\\2.fc");
//}
//double Forward(String path) {//正向传播序列
//	cout << "对 " << path << " 进行前向传播" << endl;
//	cnn1.Forward(LoadPicture(path));
//	pool1.Forward(cnn1.output_a);
//	cnn2.Forward(pool1.output);
//	pool2.Forward(cnn2.output_a);
//	cnn3.Forward(pool2.output);
//	pool3.Forward(cnn3.output_a);
//	flatted = Flatten(pool3.output);
//	fc1.Forward(flatted);
//	fc2.Forward(fc1.output_a);
//	Prediction(fc2.output_a);
//	int lable;
//	for (int i = 0; i < TOTAL_ELEMENT; i++) {
//		if (nameBuffer[i] == LabelFetch(path)) {
//			lable = i;
//			break;
//		}
//		if (i == TOTAL_ELEMENT - 1) {
//			cout << "未知的图片标签！" << endl;
//			return -1;
//		}
//	}
//	double loss;
//	loss = -(log(fc2.output_a[lable]));//交叉熵作为损失函数
//	cout << "对于标签 " << lable << " 的损失值为 " << loss << endl;
//	return loss;
//	
//
//}
//void Backward(String path) {//反向传播序列
//	int lable;
//	for (int i = 0; i < TOTAL_ELEMENT; i++) {
//		if (nameBuffer[i] == LabelFetch(path)) {
//			lable = i;
//			break;
//		}
//		if (i == TOTAL_ELEMENT - 1) {
//			cout << "未知的图片标签！" << endl;
//			return;
//		}
//	}
//
//	fc2.delta = fc2.output_z;
//	fc2.delta[lable] -= 1;
//	fc1.Backward(fc2);
//	unflatted = UnFlatten(fc1, 8, 8, 20);
//	pool3.Backward(unflatted);//注：每一层pool的delta值对应着上一层求梯度时所用的delta值
//	cnn3.Backward(pool3.delta);
//	pool2.Backward(cnn3.delta);
//	cnn2.Backward(pool2.delta);
//	pool1.Backward(cnn2.delta);
//	fc2.Grad(fc2.delta, fc1.output_a);//计算梯度
//	fc1.Grad(fc1.delta, flatted);
//	cnn3.Grad(pool3.delta, pool2.output);
//	//cnn2.Grad(pool2.delta, pool1.output);
//	//cnn1.Grad(pool1.delta, LoadPicture(path));
//}
//void Learn() {//用梯度更新权重
//	cout << "权重更新中" << endl;
//	fc1.Update(LEARNING_RATE / BATCH);
//	fc2.Update(LEARNING_RATE / BATCH);
//	cnn1.Update(LEARNING_RATE / BATCH);
//	cnn2.Update(LEARNING_RATE / BATCH);
//	cnn3.Update(LEARNING_RATE / BATCH);
//
//}
//vector<Mat> LoadPicture(String path) {
//	vector<Mat>vessel;
//	vessel.push_back(imread(path));
//	bitwise_not(vessel[0], vessel[0]);
//	if (vessel[0].type() != CV_64FC1) {
//		for (int i = 0; i < vessel.size(); i++) {
//			cvtColor(vessel[i], vessel[i], COLOR_RGB2GRAY);//将输入的所有转换为灰度图像
//			vessel[i].convertTo(vessel[i], CV_64FC1);
//			vessel[i] /= 255.0;//归一化
//			//cout << input[i].type() << endl;
//		}
//		cout << "已完成灰度化" << endl;
//	}
//	resize(vessel[0], vessel[0], Size(64, 64));
//	return vessel;
//}
//void OutputImg() {
//
//
//	//OutputPrint(cnn1.output_a, "pic\\output\\cnn1_output_");
//OutputPrint(pool1.output, "pics\\output\\pool1_output_");
////OutputPrint(cnn2.output_a, "pics\\output\\cnn2_output_");
//OutputPrint(pool2.output, "pics\\output\\pool2_output_");
////OutputPrint(cnn3.output_a, "pics\\output\\cnn3_output_");
//OutputPrint(pool3.output, "pics\\output\\pool3_output_");
//cout << "卷积中间结果已保存至pics\\output\\" << endl;
//}