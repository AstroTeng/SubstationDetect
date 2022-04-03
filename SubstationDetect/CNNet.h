#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>
#include <ctime>
#include <random>
#include <fstream>
#define LAYER_CONV 0
#define LAYER_POOL 1
#define LAYER_ALL 2
//待处理图像应该为CV_8UC1：8比特无符号（0-255）单通道图像

class CNNLayer {//卷积层类
public:
	//int inLength;//输入图像长度
	int convLength;//卷积核长度
	int inChannel;//输入通道数
	int outChannel;//输出通道数
	std::vector<std::vector<cv::Mat>> convs;//卷积核
	std::vector<std::vector<cv::Mat>> grad_convs;//卷积核的梯度
	std::vector<double> bias;//偏置
	std::vector<double> grad_bias;//偏置的梯度
	std::vector<cv::Mat>output_z;//激活前的输出
	std::vector<cv::Mat>output_a;//激活后的输出
	std::vector<cv::Mat>delta;//反向传播时对应的delta矩阵
	CNNLayer();
	CNNLayer(int inChan, int convL, int outChan);
	std::vector<cv::Mat> Forward(std::vector<cv::Mat> lastMat);
	std::vector<cv::Mat> Backward(std::vector<cv::Mat> the_next);
	void Grad(std::vector<cv::Mat>indelta, std::vector<cv::Mat>last_act);
	void Randomization();
	void Update(double learning_rate);
	void SaveData(std::string path);
	void LoadData(std::string path);
};
class PooLayer {//池化层类
public:
	bool initialized = false;//是否已经初始化
	std::vector<cv::Mat>output;//输出
	std::vector<cv::Mat>delta;//反向传播时的delta矩阵，其长宽应该是输出矩阵对应值的翻倍
	std::vector<cv::Mat> locbuffer;//传播时对应的位置矩阵，其长和宽应该为输入图像的一半.0左上；1右上；2左下；3右下
	std::vector<cv::Mat> Forward(std::vector<cv::Mat> input);
	std::vector<cv::Mat> Backward(std::vector<cv::Mat> the_next);
	PooLayer(int inChan,int inRow,int inCol);
};
class FCLayer {//全连接层类
public:
#define ACT_RELU 0
#define ACT_SOFTMAX 1
#define ACT_NONE 2
	short int activate_type;
	int neuron_num;//神经元数量，既是输入通道数也是输出通道数
	long int last_num;//上一层中的神经元数量
	std::vector<std::vector<double>> weights;//权重
	std::vector<double> bias;//偏置
	std::vector<std::vector<double>> grad_weights;//权重梯度
	std::vector<double> grad_bias;//偏置梯度
	std::vector<double> output_z;//激活前输出
	std::vector<double> output_a;//激活后输出
	std::vector<double> delta;//损失函数对本层激活前的输出的导数
	FCLayer(int lastin,int neuron,int actype);
	std::vector<double> Forward(std::vector<double> input);//前向传播
	std::vector<double> Backward(FCLayer the_next);//反向传播，即通过下一层的信息生成本层的delta值
	void Grad(std::vector<double> indelta, std::vector<double> last_act);//生成权重和和偏置的梯度
	void Randomization();//令权重和偏置随机化
	void Update(double learning_rate);
	void SaveData(std::string path);
	void LoadData(std::string path);
};

void Prediction(std::vector<double> result);
std::vector<double> Flatten(std::vector<cv::Mat> input);//展平Mat并返回其对应的一维数组容器
std::vector<cv::Mat> UnFlatten(FCLayer the_next, int row, int col, int channel);//此过程主要为将数组形式存在的delta值还原至矩阵形式
cv::Mat Convolution(cv::Mat src, cv::Mat core, int pad_length);
cv::Mat Convolution2(cv::Mat src, cv::Mat core, int pad_length);
/*
int channel = input.size();
int row = input[0].rows;
int col = input[0].cols;
int i = index / (row * col);
int m = (index - i * row * col) / col;
int n = index - i * row * col - m * col;*/
