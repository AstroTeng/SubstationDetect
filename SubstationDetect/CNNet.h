#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>
#include <ctime>
#include <random>
#include <fstream>
#define LAYER_CONV 0
#define LAYER_POOL 1
#define LAYER_ALL 2
//������ͼ��Ӧ��ΪCV_8UC1��8�����޷��ţ�0-255����ͨ��ͼ��

class CNNLayer {//�������
public:
	//int inLength;//����ͼ�񳤶�
	int convLength;//����˳���
	int inChannel;//����ͨ����
	int outChannel;//���ͨ����
	std::vector<std::vector<cv::Mat>> convs;//�����
	std::vector<std::vector<cv::Mat>> grad_convs;//����˵��ݶ�
	std::vector<double> bias;//ƫ��
	std::vector<double> grad_bias;//ƫ�õ��ݶ�
	std::vector<cv::Mat>output_z;//����ǰ�����
	std::vector<cv::Mat>output_a;//���������
	std::vector<cv::Mat>delta;//���򴫲�ʱ��Ӧ��delta����
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
class PooLayer {//�ػ�����
public:
	bool initialized = false;//�Ƿ��Ѿ���ʼ��
	std::vector<cv::Mat>output;//���
	std::vector<cv::Mat>delta;//���򴫲�ʱ��delta�����䳤��Ӧ������������Ӧֵ�ķ���
	std::vector<cv::Mat> locbuffer;//����ʱ��Ӧ��λ�þ����䳤�Ϳ�Ӧ��Ϊ����ͼ���һ��.0���ϣ�1���ϣ�2���£�3����
	std::vector<cv::Mat> Forward(std::vector<cv::Mat> input);
	std::vector<cv::Mat> Backward(std::vector<cv::Mat> the_next);
	PooLayer(int inChan,int inRow,int inCol);
};
class FCLayer {//ȫ���Ӳ���
public:
#define ACT_RELU 0
#define ACT_SOFTMAX 1
#define ACT_NONE 2
	short int activate_type;
	int neuron_num;//��Ԫ��������������ͨ����Ҳ�����ͨ����
	long int last_num;//��һ���е���Ԫ����
	std::vector<std::vector<double>> weights;//Ȩ��
	std::vector<double> bias;//ƫ��
	std::vector<std::vector<double>> grad_weights;//Ȩ���ݶ�
	std::vector<double> grad_bias;//ƫ���ݶ�
	std::vector<double> output_z;//����ǰ���
	std::vector<double> output_a;//��������
	std::vector<double> delta;//��ʧ�����Ա��㼤��ǰ������ĵ���
	FCLayer(int lastin,int neuron,int actype);
	std::vector<double> Forward(std::vector<double> input);//ǰ�򴫲�
	std::vector<double> Backward(FCLayer the_next);//���򴫲�����ͨ����һ�����Ϣ���ɱ����deltaֵ
	void Grad(std::vector<double> indelta, std::vector<double> last_act);//����Ȩ�غͺ�ƫ�õ��ݶ�
	void Randomization();//��Ȩ�غ�ƫ�������
	void Update(double learning_rate);
	void SaveData(std::string path);
	void LoadData(std::string path);
};

void Prediction(std::vector<double> result);
std::vector<double> Flatten(std::vector<cv::Mat> input);//չƽMat���������Ӧ��һά��������
std::vector<cv::Mat> UnFlatten(FCLayer the_next, int row, int col, int channel);//�˹�����ҪΪ��������ʽ���ڵ�deltaֵ��ԭ��������ʽ
cv::Mat Convolution(cv::Mat src, cv::Mat core, int pad_length);
cv::Mat Convolution2(cv::Mat src, cv::Mat core, int pad_length);
/*
int channel = input.size();
int row = input[0].rows;
int col = input[0].cols;
int i = index / (row * col);
int m = (index - i * row * col) / col;
int n = index - i * row * col - m * col;*/
