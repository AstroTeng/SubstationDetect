#pragma once
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;
#define PI 3.14159
#define DIVIDE(A) canvaRow/(A*1.0)
#define TOTAL_ELEMENT 8
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
Mat RandomPic(Mat in);//对图像进行一定的随机变换
class ElementGenerate{
public:
	ElementGenerate(int row, int col, String loc);
	ElementGenerate();
	void ShowInfo();
	void SetValue(int row, int col, String loc);

	void Transformer3P(RNG rng);
	void Transformer(RNG rng);
	void IsoSwitch(RNG rng);
	void Breaker(RNG rng);
	void Load(RNG rng);
	void Ground(RNG rng);
	void PureLine(RNG rng);
private:
	void WriteImg(int eleNo, Mat& target);
	void CanvaSizeRandom(RNG random_core);
	String saveLocation;
	int canvaRow;
	int canvaRow_o;
	int canvaCol;
	int canvaCol_o;
	int eleBuffer[TOTAL_ELEMENT];//每一个元素生成的总数
	int total_num;//总共生成的图像总数
	default_random_engine re;
};
void PathFetch(vector<String>& targetVessel, String fileType, String addPath);
void ImgBinar(Mat& inImg);
void DeltaGene(Point core, Point* vertexArray, double radius);
Point PointAngel(Point core, double radius, double angle);
void RandomRotate(Mat& inImg, RNG rng);
void RandomScaling(Mat& inImg, RNG rng);
static String nameBuffer[TOTAL_ELEMENT] = {
	"0",//三相变压器
	"1",//变压器
	"2",//隔离开关
	"3",//断路器
	"4",//负载
	"5"//接地
};

string LabelFetch(string path);

