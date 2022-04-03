#include "ElementGenerate.h"

Mat RandomPic(Mat in) {
	default_random_engine e(getTickCount());
	uniform_real_distribution<double> up(0, in.rows * 0.1);
	uniform_real_distribution<double> down(0, in.rows * 0.1);
	uniform_real_distribution<double> left(0, in.cols * 0.1);
	uniform_real_distribution<double> right(0, in.cols * 0.1);
	Mat out = in.clone();
	copyMakeBorder(out, out, up(e), down(e), left(e), right(e), BORDER_CONSTANT, Scalar(255, 255, 255));
	resize(out, out, in.size());
	return out;

}
ElementGenerate::ElementGenerate(void) {
	saveLocation = "pics/";
	canvaRow_o=canvaRow = 64;
	canvaCol_o=canvaCol = 64;
	total_num = 10000;
	for (int i = 0; i < TOTAL_ELEMENT; i++)
		eleBuffer[i] = 0;
	ShowInfo();
}
ElementGenerate::ElementGenerate(int row, int col, String loc) {
	default_random_engine e(getTickCount());
	re = e;
	SetValue(row, col, loc);
	total_num = 10000;
} 
void ElementGenerate::ShowInfo(void) {
	cout << "Drawing on a" << this->canvaRow << " * " << this->canvaCol << " canva at the location:" << this->saveLocation << endl;
	return;
}
void ElementGenerate::SetValue(int row, int col, String loc) {
	saveLocation = loc;
	canvaRow_o = canvaRow = row;
	canvaCol_o = canvaCol = col;
	for (int i = 0; i < TOTAL_ELEMENT; i++)
		eleBuffer[i] = 0;
	ShowInfo();
	return;
}
void ElementGenerate::CanvaSizeRandom(RNG random_core) {
	uniform_real_distribution<double> dis(0.28, 1);
	//double scale = random_core.uniform(0.4, 1.1);
	double scale = dis(re);
	canvaCol = canvaCol_o * scale;
	canvaRow = canvaRow_o * scale;
	cout << "将画在" << canvaRow << " ×" << canvaCol << "的画布上" << endl;
}
void ElementGenerate::PureLine(RNG rng) {
	CanvaSizeRandom(rng);
	uniform_real_distribution<double> dis1(0, this->canvaRow);
	uniform_real_distribution<double> dis2(0, this->canvaRow);
	uniform_int_distribution<int> dis3(1, 3);
	uniform_int_distribution<int> dis4(1, 2);
	Point main_point(dis1(re), dis2(re));
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	int thick = dis4(re);
	switch (dis3(re)) {
	case 3:
		line(canv, main_point, Point(main_point.x, canvaRow), Scalar(0), thick);
	case 2:
		line(canv, main_point, Point(0, main_point.y), Scalar(0), thick);
	case 1: 
		line(canv, main_point, Point(main_point.x, 0), Scalar(0),thick);
		break;
	}
	RandomRotate(canv, rng);
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	String fileName = to_string(total_num++) +//总数
		"_" + "0" +
		"_x" + to_string(canvaRow_o) +
		".jpg";
	fileName[0] = 'n';
	String path = saveLocation;
	path += fileName;
	cout << "在以下路径生成：" << path << endl;
	imwrite(path, canv);

}
void ElementGenerate::Transformer3P(RNG rng) {//三相变压器
	CanvaSizeRandom(rng);
	Point core(DIVIDE(2), DIVIDE(2));
	Point vertex[3];
	uniform_real_distribution<double> dis(DIVIDE(6) * 1, DIVIDE(3) * 0.6);
	//int R = rng.uniform(DIVIDE(6)*1.2, DIVIDE(3)*0.8);
	int R = dis(re);
	int r = DIVIDE(2) - R;
	int thickness = rng.uniform(1, 3);
	DeltaGene(core, vertex, R);
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	circle(canv, vertex[0], r, Scalar(0), thickness);
	circle(canv, vertex[1], r, Scalar(0), thickness);
	circle(canv, vertex[2], r, Scalar(0), thickness);
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	RandomRotate(canv, rng);
	WriteImg(0, canv);
	return;
}
void ElementGenerate::Transformer(RNG rng) {//两相变压器
	CanvaSizeRandom(rng);
	Point core1, core2;
	int coreToE = rng.uniform(DIVIDE(4) * 1.1, DIVIDE(4) * 1.55);
	int thickness = rng.uniform(1, 3);
	int radius = coreToE - rng.uniform(0, 2);
	core1.x = canvaRow / 2;
	core1.y = coreToE;
	core2.x = core1.x;
	core2.y = canvaRow - coreToE;
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	circle(canv, core1, radius, Scalar(0), thickness);
	circle(canv, core2, radius, Scalar(0), thickness);
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	WriteImg(1, canv);
	return;
}
void ElementGenerate::IsoSwitch(RNG rng) {//隔离开关
	CanvaSizeRandom(rng);
	int headToE = rng.uniform(DIVIDE(3) * 0.7, DIVIDE(3) * 1);
	int buttonToE = canvaRow - rng.uniform(DIVIDE(3) * 0.4, DIVIDE(3) * 0.7);
	int upToE = rng.uniform(DIVIDE(2) * 0.7, DIVIDE(2) * 0.8);
	int thickness = rng.uniform(1, 3);
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	line(canv, Point(DIVIDE(2), headToE), Point(DIVIDE(2), 0), Scalar(0), thickness);//|
	line(canv, Point(upToE, headToE), Point(canvaCol - upToE, headToE), Scalar(0), thickness);//-
	line(canv, Point(DIVIDE(2), buttonToE), Point(DIVIDE(2), canvaRow), Scalar(0), thickness);//|
	line(canv, Point(DIVIDE(2), buttonToE), Point(upToE * rng.uniform(0.5, 0.8), headToE * rng.uniform(1.0, 1.1)), Scalar(0), thickness);// 斜线
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	RandomRotate(canv, rng);
	WriteImg(2, canv);

}
void ElementGenerate::Breaker(RNG rng) {//断路器
	CanvaSizeRandom(rng);
	int headToE = rng.uniform(DIVIDE(3) * 0.7, DIVIDE(3) * 1);
	int buttonToE = canvaRow - rng.uniform(DIVIDE(3) * 0.4, DIVIDE(3) * 0.7);
	int upToE = rng.uniform(DIVIDE(2) * 0.7, DIVIDE(2) * 0.8);
	int halfSquare = (DIVIDE(2) - upToE) * rng.uniform(0.5, 0.8);
	Point lu(DIVIDE(2) - halfSquare, headToE - halfSquare), 
		ru(DIVIDE(2) + halfSquare, headToE - halfSquare),
		ld(DIVIDE(2) - halfSquare, headToE + halfSquare), 
		rd(DIVIDE(2) + halfSquare, headToE + halfSquare);
	int thickness = rng.uniform(1, 3);
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	line(canv, Point(DIVIDE(2), headToE), Point(DIVIDE(2), 0), Scalar(0), thickness);//|
	line(canv, lu, rd, Scalar(0), thickness);//叉叉
	line(canv, ld, ru, Scalar(0), thickness);//叉叉
	line(canv, Point(DIVIDE(2), buttonToE), Point(DIVIDE(2), canvaRow), Scalar(0), thickness);//|
	line(canv, Point(DIVIDE(2), buttonToE), Point(upToE * rng.uniform(0.5, 0.8), headToE * rng.uniform(1.0, 1.1)), Scalar(0), thickness);// 斜线
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	RandomRotate(canv,rng);
	WriteImg(3, canv);

}
void ElementGenerate::Load(RNG rng) {//负载
	CanvaSizeRandom(rng);
	int buttonToE = canvaRow - rng.uniform(DIVIDE(1) * 0.1, DIVIDE(1) * 0.6);
	double slapLength = rng.uniform(DIVIDE(3) * 0.5, DIVIDE(3));
	double angle = rng.uniform(5, 45);
	int thickness = rng.uniform(1, 3);
	Point core(DIVIDE(2), buttonToE);
	Point vertex[3];
	vertex[0] = core;
	vertex[1] = PointAngel(core, slapLength, angle);
	vertex[2] = PointAngel(core, slapLength, -angle);
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	line(canv, Point(DIVIDE(2), 0), Point(DIVIDE(2), buttonToE), Scalar(0), thickness);
	switch (rng.uniform(0, 2)) {
	default:
	case 0://箭头
		line(canv, vertex[0], vertex[1], Scalar(0), thickness);
		line(canv, vertex[0], vertex[2], Scalar(0), thickness);
		break;
	case 1://三角形
		fillConvexPoly(canv, vertex, 3, Scalar(0));
		break;
	}
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	RandomRotate(canv, rng);
	WriteImg(4, canv);
	return;
	
}
void ElementGenerate::Ground(RNG rng) {//接地
	CanvaSizeRandom(rng);
	int headToE = rng.uniform(DIVIDE(3) * 0.7, DIVIDE(3) * 1.2);
	int buttonToE = canvaRow - rng.uniform(DIVIDE(3) * 0.95, DIVIDE(3) * 1.2);
	int upToE = rng.uniform(DIVIDE(5) * 0.9, DIVIDE(5) * 1.1);
	int downToE = rng.uniform(DIVIDE(2) * 0.75, DIVIDE(2) * 0.9);
	int thickness = rng.uniform(1, 3);
	Mat canv(canvaRow, canvaCol, CV_8UC1, Scalar(255));
	line(canv, Point(DIVIDE(2), headToE), Point(DIVIDE(2), 0), Scalar(0), thickness);
	line(canv, Point(upToE, headToE), Point(canvaCol-upToE, headToE), Scalar(0), thickness);
	line(canv, Point(downToE, buttonToE), Point(canvaCol - downToE, buttonToE), Scalar(0), thickness);
	line(canv, Point((downToE + upToE) / 2, (buttonToE + headToE) / 2), Point((canvaCol - downToE + canvaCol - upToE) / 2, (buttonToE + headToE) / 2), Scalar(0), thickness);
	resize(canv, canv, Size(canvaRow_o, canvaCol_o));
	WriteImg(5, canv);
}

void ElementGenerate::WriteImg(int eleNo, Mat& target) {//在指定路径生成jpg图像文件

	String fileName = to_string(total_num++) +//总数
		"_" + nameBuffer[eleNo] +
		"_x" + to_string(canvaRow_o) +
		".jpg";
	fileName[0] = 'n';
	String path = saveLocation;
	path += fileName;
	cout << "在以下路径生成：" << path << endl;
	imwrite(path, target);
}
void PathFetch(vector<String>& targetVessel, String fileType, String Path) {//从工程目录下的pics文件夹中读取指定类型的图像到指定容器中
	String rootPath =Path;
	rootPath +=   "*." + fileType;
	cout << "寻找所有在以下目录的文件： \"" << rootPath << "\"" << endl;
	glob(rootPath, targetVessel, false);//recusive递归
	if (targetVessel.size() == 0) {
		cout << "文件夹中无该文件类型！" << endl;
		return;
	}
	cout << "已找到" << targetVessel.size() << " [" << fileType << "]:" << endl;
	for (vector<String>::iterator ptr = targetVessel.begin(); ptr != targetVessel.end(); ptr++)
		cout << ptr - targetVessel.begin() << ". " << *ptr << endl;
	cout << endl;
	return;
}
void ImgBinar(Mat& inImg) {//图像二值化
	threshold(inImg, inImg, 200, 255, THRESH_BINARY);
	return;
}
void DeltaGene(Point core, Point* vertexArray, double radius) {//往Point数组中填入正三角形的三个顶点
	vertexArray[0] = PointAngel(core, radius, 0);
	vertexArray[1] = PointAngel(core, radius, 120);
	vertexArray[2] = PointAngel(core, radius, 240);
		return;
}
Point PointAngel(Point core, double radius, double angle) {//以core所在位置为原点，在距离竖直方向angle角度且距离core radius的位置生成一个新的点
	Point result;
	result.x = core.x + radius * sin(angle / 180 * PI);
	result.y = core.y - radius * cos(angle / 180 * PI);
	return result;
}
void RandomRotate(Mat& inImg,RNG rng) {//将传入的图像随机旋转90、180或者270度
	switch (rng.uniform(0, 4)) {
	default:
	case 0:
		break;
	case 1:
		cv::rotate(inImg, inImg, ROTATE_90_CLOCKWISE);
		break;
	case 2:
		cv::rotate(inImg, inImg, ROTATE_180);
		break;
	case 3:
		cv::rotate(inImg, inImg, ROTATE_90_COUNTERCLOCKWISE);
		break;
	}
}
void RandomScaling(Mat& inImg, RNG rng) {//将传入的图像随机放缩后再放缩一次到指定输出尺寸
	int origin_col = inImg.cols;
	int origin_row = inImg.rows;
	double scaleRatio = rng.uniform(0.4, 1.0);
	resize(inImg, inImg, Size(0, 0), scaleRatio, scaleRatio, INTER_LINEAR);
	resize(inImg, inImg, Size(origin_col, origin_row));
}
string LabelFetch(string path) {//获取标签
	int i = 0;
	while (path[i] != '_') i++;
	i++;
	int n = 0;
	for (int j = i; path[j] != '_'; j++) n++;
	return path.substr(i, n);
}