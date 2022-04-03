#include "CNNet.h"
//#include "ElementGenerate.h"

using namespace cv;
using namespace std;

CNNLayer::CNNLayer() {//默认构造
	this->convLength = 3;
	this->inChannel = 1;
	this->outChannel = 10;
	for (int i = 0; i < outChannel; i++) {
		vector<Mat> a_row;
		for (int j = 0; j < inChannel; j++) {
			a_row.push_back(Mat::zeros(convLength, convLength, CV_64FC1));
		}
		this->convs.push_back(a_row);
		this->bias.push_back(0);
	}
}
CNNLayer::CNNLayer(int inChan,int convL,int outChan) {//输入通道数、卷积核长度、输出通道数
	if (convL > 0 && convL % 2 != 0) {//确保卷积核长度为奇数
		this->convLength = convL;
		this->outChannel = outChan;
		this->inChannel = inChan;
		for (int i = 0; i < outChan; i++) {
			vector<Mat> a_row;
			for (int j = 0; j < inChan; j++) {
				a_row.push_back(Mat::zeros(convLength, convLength, CV_64FC1));
			}
			this->convs.push_back(a_row);
			this->grad_convs.push_back(a_row);
			this->bias.push_back(0);
			this->grad_bias.push_back(0);
			Mat tmp;
			output_z.push_back(tmp);
			output_a.push_back(tmp);
		}
		for (int i = 0; i < inChannel; i++)
			delta.push_back(Mat::zeros(1, 1, CV_64FC1));
		//grad_convs = convs;//不可以这样！这是浅复制！
		//grad_bias = bias;
		cout << "卷积层已创建" << endl;
	}
	else cout << "失败！卷积核的长度一定为奇数！" << endl;
}
vector<Mat> CNNLayer::Forward(vector<Mat> input) {//卷积层前向传播操作，结果将储存于vector<Mat>output中
	cout << "\033[33m[卷积层：前向传播]\033[0m" << endl;
	if (input.size() != inChannel) {
		cout << "失败！输入通道数与该层不匹配！" << endl;
		return output_z;
	}
	time_t begin, end;
	double ret;
	begin = clock();
	if (input[0].type() != CV_64FC1) {
		for (int i = 0; i < input.size(); i++) {
			cvtColor(input[i], input[i], COLOR_RGB2GRAY);//将输入的所有转换为灰度图像
			input[i].convertTo(input[i], CV_64FC1);
			input[i] /= 255.0;//归一化
			//cout << input[i].type() << endl;
		}
		cout << "已完成灰度化" << endl;
	}
	for (int i = 0; i < outChannel; i++) {
		//Mat sub;
		for (int j = 0; j < inChannel; j++) {
			Mat tmp;
			filter2D(input[j], tmp, CV_64FC1, convs[i][j], Point(-1, -1), 0, BORDER_CONSTANT);
			if (j == 0) {
				output_z[i] = tmp;
				//sub = Convolution(input[j], convs[i][j], convLength / 2);
			}
			else {
				output_z[i] += tmp;
				//sub += Convolution(input[j], convs[i][j], convLength / 2);
			}
			/*for (int m = 0; m < sub.rows; m++)//测试模块
				for (int n = 0; n < sub.rows; n++)
					if (sub.at<double>(m, n) != output_z[i].at<double>(m, n)) {
						cout << "output_z:" << output_z[i].at<double>(m, n) << endl;
						cout << "sub:" << sub.at<double>(m, n) << endl;
						Mat inex;
						copyMakeBorder(input[j], inex,
							convLength / 2, convLength / 2, convLength / 2, convLength / 2,
							BORDER_CONSTANT, 0);
						cout << "input[" << j << "]" << endl;
						for(int o=m;o<m+convLength;o++)
							for (int p = n; o < n + convLength; p++) {
								cout << inex.at<double>(m, n) << ",";
								if (p == n + convLength - 1)
									cout << endl;
							}
						cout << "convs[" << i << "][" << j << "]:" << endl;
						cout << convs[i][j] << endl;
					}*/
		}
		output_z[i] += bias[i];
		//sub += bias[i];
		output_a[i] = output_z[i] ;
		for (int m = 0; m < output_a[i].rows; m++)//relu激活函数
			for (int n = 0; n < output_a[i].cols; n++)
				if (output_a[i].at<double>(m, n) < 0)
					output_a[i].at<double>(m, n) = 0;
	}
	end = clock();
	ret = double(end - begin) * 1000 / CLOCKS_PER_SEC;
	cout << "\033[32m前向传播完成！\033[0m" << endl;
	cout << "输入：" << input.size() << " 通道；" ;
	cout << "输出：" << output_a.size() << " 通道；" ;
	cout << "特征图长度：" << input[0].rows << endl;
	return output_a;
}
vector<Mat> CNNLayer::Backward(vector<Mat> the_next) {//卷积层的delta反向传播
	cout << "\033[33m[卷积层：反向传播]\033[0m" << endl;
	time_t begin, end;
	double ret;
	begin = clock();
	for (int i = 0; i < inChannel; i++) {
		for (int j = 0; j < outChannel; j++) {
			Mat conv_tmp;
			Mat tmp;
			flip(convs[j][i], conv_tmp, 0);//卷积矩阵翻转180°
			flip(conv_tmp, conv_tmp, 1);
			filter2D(the_next[j], tmp, CV_64FC1, conv_tmp, Point(-1, -1), 0, BORDER_CONSTANT);//卷积
			for (int m = 0; m < tmp.rows; m++)//相当于和激活函数的导数相乘
				for (int n = 0; n < tmp.cols; n++)
					if (output_z[j].at<double>(m, n) < 0)
						tmp.at<double>(m, n) = 0;
			if (j == 0)
				delta[i] = tmp;
			else
				delta[i] += tmp;
		}
	}

	end = clock();
	ret = double(end - begin) * 1000 / CLOCKS_PER_SEC;
	cout << ret << "ms" << endl;

	return delta;
}
void CNNLayer::Grad(std::vector<cv::Mat>indelta, std::vector<cv::Mat>last_act) {
	cout << "\033[33m[卷积层：计算梯度]\033[0m" << endl;
	//cout << "grad_convs:" << grad_convs.size() << "×" << grad_convs[0].size() << endl;
	//cout << "indelta.size():" << indelta.size() << endl;
	//cout << "last_act.size()" << last_act.size() << endl;
	//计算卷积权重梯度
	for (int i = 0; i < indelta.size(); i++)
		for (int j = 0; j < last_act.size(); j++){
			//Mat sub;
			grad_convs[i][j] += Convolution2(last_act[j], indelta[i], convLength / 2);
			//sub=Convolution(last_act[j], indelta[i], convLength / 2);
			//cout << "grad_convs[" << i << "][" << j << "]:done" << endl;
			//cout << grad_convs[i][j] << endl;
			//cout << sub << endl;
		}

	//计算偏置梯度
	for (int i = 0; i < bias.size(); i++) {
		for (int m = 0; m < indelta[i].rows; m++)
			for (int n = 0; n < indelta[i].cols; n++)
				grad_bias[i] += indelta[i].at<double>(m, n);
	}
	cout << "完成" << endl;
}
void CNNLayer::Randomization() {//随机化权重和偏置
	default_random_engine e{ (unsigned int)time(NULL) };//http://c.biancheng.net/view/638.html C++随机数生成器（default_random_engine）用法详解
	double std = 2.0 / (inChannel * convLength* convLength);
	normal_distribution<double> d(0, std);
	double sum = 0;
	cout << "插入权重：";
	for (int i = 0; i < outChannel; i++) {
		for(int j=0;j<inChannel;j++)
			for (int m = 0; m < convs[i][j].cols; m++)
				for (int n = 0; n < convs[i][j].rows; n++) {
					convs[i][j].at<double>(m, n) = d(e);
					//cout << convs[i][j].at<double>(m, n) << " ";
				}
		bias[i] = 0;
	}
	return;
}
void CNNLayer::Update(double learning_rate) {
	cout << "\033[33m[卷积层：权重更新]\033[0m" << endl;
	for (int i = 0; i < grad_convs.size(); i++)
		for (int j = 0; j < grad_convs[0].size(); j++) {
			//cout<< convs[i][j] <<endl;
			//cout << grad_convs[i][j] << endl;
			convs[i][j] = convs[i][j] - learning_rate * grad_convs[i][j];
			grad_convs[i][j] *= 0;
		}

	for (int i = 0; i < grad_bias.size(); i++) {
		bias[i] -= learning_rate * grad_bias[i];
		grad_bias[i] = 0;
	}
	cout << "完成" << endl;
}
void FCLayer::Update(double learning_rate) {
	cout << "\033[33m[全连接层：权重更新]\033[0m" << endl;
	for (int i = 0; i < weights.size(); i++)
		for (int j = 0; j < weights[0].size(); j++) {
			weights[i][j] -= learning_rate * grad_weights[i][j];
			grad_weights[i][j] = 0;
		}

	for (int i = 0; i < grad_bias.size(); i++) {
		bias[i] -= learning_rate * grad_bias[i];
		grad_bias[i] = 0;
	}

	cout << "完成" << endl;
}
void CNNLayer::SaveData(string path) {//保存数据
	ofstream out;
	cout << "卷积层：保存数据至" << path << " ...";
	out.open(path, ios::binary);
	out.write((char*)&convLength, sizeof(int));
	out.write((char*)&inChannel, sizeof(int));
	out.write((char*)&outChannel, sizeof(int));
	for (int i = 0; i < outChannel; i++) {
		for (int j = 0; j < inChannel; j++) {
			for (int m = 0; m < convs[i][j].cols; m++)
				for (int n = 0; n < convs[i][j].rows; n++)
					out.write((char*)&convs[i][j].at<double>(m, n), sizeof(double));
		}
		out.write((char*)&bias[i], sizeof(double));
	}
	out.close();
	cout << "完成！" << endl;
}
void CNNLayer::LoadData(string path) {//读取数据
	ifstream in;
	cout << "卷积层：匹配数据格式...";
	int info_array[3];
	in.open(path, ios::binary);
	in.read((char*)&info_array[0], sizeof(int));
	in.read((char*)&info_array[1], sizeof(int));
	in.read((char*)&info_array[2], sizeof(int));
	if (!(info_array[0] == convLength && info_array[1] == inChannel && info_array[2] == outChannel)) {
		cout << "数据不匹配！" << endl;
		return;
	}
	cout << "成功！" << endl;
	cout << "从 " << path << " 读取数据...";
	for (int i = 0; i < outChannel; i++) {
		for (int j = 0; j < inChannel; j++) {
			for (int m = 0; m < convs[i][j].cols; m++)
				for (int n = 0; n < convs[i][j].rows; n++)
					in.read((char*)&convs[i][j].at<double>(m, n), sizeof(double));
		}
		in.read((char*)&bias[i], sizeof(double));
	}
	in.close();
	cout << "完成！" << endl;
}

PooLayer::PooLayer(int inChan, int inRow, int inCol) {//池化层对象的构造函数
	for (int i = 0; i < inChan; i++) {
		output.push_back(Mat::zeros(inRow / 2, inCol / 2, CV_64FC1));
		delta.push_back(Mat::zeros(inRow, inCol, CV_64FC1));
		locbuffer.push_back(Mat::zeros(inRow / 2, inCol / 2, CV_8UC1));
	}
	cout << "池化层已创建" << endl;
}
vector<Mat> PooLayer::Forward(vector<Mat> input) {//池化层前向传播操作
	cout << "\033[33m[池化层：前向传播]\033[0m" << endl;
	if (input[0].cols % 2 || input[0].rows % 2) {
		cout << "失败！输入矩阵的长度必须为偶数！" << endl;
		return output;
	}
	else if (input.size() != output.size()) {
		cout << "失败！输入通道数不等于输出通道数！" << endl;
		return output;
	}
	for (int i = 0; i < output.size(); i++) {
		for (int m = 0; m < input[i].rows; m = m + 2)
			for (int n = 0; n < input[i].cols; n = n + 2) {
				double max = input[i].at<double>(m, n);
				if (input[i].at<double>(m, n + 1) > max) {
					max = input[i].at<double>(m, n + 1);
					locbuffer[i].at<unsigned char>(m / 2, n / 2) = 1;
				}
				if (input[i].at<double>(m + 1, n) > max) {
					max = input[i].at<double>(m + 1, n);
					locbuffer[i].at<unsigned char>(m / 2, n / 2) = 2;
				}
				if (input[i].at<double>(m + 1, n + 1) > max) {
					max = input[i].at<double>(m + 1, n + 1);
					locbuffer[i].at<unsigned char>(m / 2, n / 2) = 3;
				}
				output[i].at<double>(m / 2, n / 2) = max;
			}
	}
	cout << "\033[32m前向传播完成！\033[0m" << endl;
	cout << "输入/输出通道数：" << input.size();
	cout << "；输入矩阵长度：" << input[0].rows;
	cout << "；输出矩阵长度：" << output[0].rows << endl;
	return output;
}
vector<Mat> PooLayer::Backward(vector<Mat> the_next) {
	cout << "\033[33m[池化层：反向传播]\033[0m" << endl;
	time_t begin, end;
	double ret;
	begin = clock();
	for (int i = 0; i < delta.size(); i++) {
		for (int m = 0; m < delta[i].rows; m = m + 2) {
			for (int n = 0; n < delta[i].cols; n = n + 2) {
				switch (locbuffer[i].at<unsigned char>(m / 2, n / 2)) {
				case 0:
					delta[i].at<double>(m, n) = the_next[i].at<double>(m / 2, n / 2);
					break;
				case 1:
					delta[i].at<double>(m, n + 1) = the_next[i].at<double>(m / 2, n / 2);
					break;
				case 2:
					delta[i].at<double>(m + 1, n) = the_next[i].at<double>(m / 2, n / 2);
					break;
				case 3:
					delta[i].at<double>(m + 1, n + 1) = the_next[i].at<double>(m / 2, n / 2);
					break;
				}
			}
		}
	}
	end = clock();
	ret = double(end - begin) * 1000 / CLOCKS_PER_SEC;
	//cout << ret << "ms" << endl;
	return delta;
}
FCLayer::FCLayer(int lastin, int neuron, int actype) {
	this->neuron_num = neuron;
	activate_type = actype;
	last_num = lastin;
	for (int i = 0; i < neuron_num; i++) {
		vector<double> w;
		for (int j = 0; j < last_num; j++) {
			w.push_back(0);
		}
		weights.push_back(w);
		bias.push_back(0);
		output_z.push_back(0);
		output_a.push_back(0);
	}
	grad_weights = weights;
	grad_bias = bias;
	delta = output_z;
	cout << "全连接层已创建" << endl;
}
void FCLayer::Randomization() {
	default_random_engine e;
	double std = 2.0 / neuron_num;
	switch (activate_type) {//根据不同的激活函数来选择不同的方差
	default:
	case ACT_RELU:
		std = 2.0 / neuron_num;
		break;
	case ACT_SOFTMAX:
		std = 2.0 / (neuron_num);
	}
	normal_distribution<double> d(0, std);//正态分布初始化，左边为均值，右边为方差
	for (int i = 0; i < neuron_num; i++) {
		for (int j = 0; j < last_num; j++) 
			weights[i][j] = d(e);
		bias[i]=0;
	}
	return;
}
void FCLayer::SaveData(string path) {//保存数据
	ofstream out;
	cout << "全连接层：保存数据至 "<<path<<" ...";
	out.open(path, ios::binary);
	out.write((char*)&last_num, sizeof(int));
	out.write((char*)&neuron_num, sizeof(int));
	out.write((char*)&activate_type, sizeof(int));
	for (int i = 0; i < neuron_num; i++) {
		for (int j = 0; j < last_num; j++) 
			out.write((char*)&weights[i][j], sizeof(double));
		out.write((char*)&bias[i], sizeof(double));
	}
	out.close();
	cout << "完成！" << endl;
}
void FCLayer::LoadData(string path) {
	ifstream in;
	cout << "全连接层：匹配数据格式...";
	int info_array[3];
	in.open(path, ios::binary);
	in.read((char*)&info_array[0], sizeof(int));
	in.read((char*)&info_array[1], sizeof(int));
	in.read((char*)&info_array[2], sizeof(int));
	//printf("Read: convLength: %d inChannel: %d outChannel: %d\n",
		//info_array[0], info_array[1], info_array[2]);
	if (!(info_array[0] == last_num && info_array[1] == neuron_num && info_array[2] == activate_type)) {
		cout << "数据格式不匹配！" << endl;
		return;
	}
	cout << "成功！" << endl;
	cout << "从 " << path << " 读取数据...";
	for (int i = 0; i < neuron_num; i++) {
		for (int j = 0; j < last_num; j++)
			in.read((char*)&weights[i][j], sizeof(double));
		in.read((char*)&bias[i], sizeof(double));
	}
	in.close();
	cout << "完成！" << endl;
}
vector<double> FCLayer::Forward(vector<double> input) {
	cout << "\033[33m[全连接层：前向传播]\033[0m" << endl;
	time_t begin, end;
	double ret;
	begin = clock();
	for (int i = 0; i < neuron_num; i++) {
		double sum = 0;
		for (int j = 0; j < last_num; j++) 
			sum += input[j] * weights[i][j];
		sum += bias[i];
		output_z[i] = output_a[i] = sum;
	}
	double expsum = 0;
	cout << "激活函数种类：";
	switch (activate_type) {//7
	default:
	case ACT_RELU:
		cout << "Relu" << endl;
		for (int i = 0; i < output_a.size(); i++)
			if (output_a[i] < 0) output_a[i] = 0;
		break;
	case ACT_SOFTMAX:
		cout << "Softmax" << endl;
		for (int i = 0; i < output_a.size(); i++) {
			output_a[i] = exp(output_a[i]);
			expsum += output_a[i];
		}
		for (int i = 0; i < output_a.size(); i++)
			output_a[i] /= expsum;
		break;
	case ACT_NONE:
		cout << "无" << endl;
		break;
	}
	end = clock();
	ret = double(end - begin) * 1000 / CLOCKS_PER_SEC;
	cout << "\033[32m前向传播完成！\033[0m" << endl;
	cout << "输入大小：" << input.size() ;
	cout << "；输出大小：" << output_a.size() << endl;
	return output_a;
}
vector<double> FCLayer::Backward(FCLayer the_next) {
	cout << "\033[33m[全连接层：反向传播]\033[0m" << endl;
	for (int i = 0; i < the_next.last_num; i++) {//和前向传播相比，此处的矩阵转置了
		double sum = 0;
		for (int j = 0; j < the_next.neuron_num; j++) {
			sum += the_next.delta[j] * the_next.weights[j][i];
		}
		delta[i] = sum;
		if (output_z[i] < 0)
			delta[i] = 0;
	}
	cout << "完成" << endl;
	return delta;
}
void FCLayer::Grad(vector<double> indelta, vector<double> last_act) {
	cout << "\033[33m[全连接层：计算梯度]\033[0m" << endl;
	for (int i = 0; i < indelta.size(); i++) {
		grad_bias[i] += indelta[i];//偏置梯度
		for (int j = 0; j < last_act.size(); j++)
			grad_weights[i][j] += indelta[i] * last_act[j];//权重梯度
	}
	cout << "完成" << endl;
}
vector<double> Flatten(vector<Mat> input) {
	cout << "\033[33m[矩阵展平]\033[0m" << endl;
	vector<double> output;
	int channel = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	for (long int index = 0; index < channel * row * col; index++) {
		int i = index / (row * col);
		int m = (index - i * row * col) / col;
		int n = index - i * row * col - m * col;
		output.push_back(input[i].at<double>(m, n));
	}
	cout << "\033[32m展平完成！\033[0m" << endl;
	cout << "输入：" << row << " ×" << col << " ×" << input.size() ;
	cout << "；输出：" << output.size() << endl;
	return output;
}
vector<Mat> UnFlatten(FCLayer the_next,int row,int col,int channel) {//此过程主要为将数组形式存在的delta值还原至矩阵形式
	cout << "\033[33m[数组回归矩阵]\033[0m" << endl;
	vector<Mat> delta;
	vector<double> delta_linear;
	for (int i = 0; i < the_next.last_num; i++) {//和前向传播相比，此处的矩阵转置了
		double sum = 0;
		for (int j = 0; j < the_next.neuron_num; j++) {
			sum += the_next.delta[j] * the_next.weights[j][i];
		}
		delta_linear.push_back(sum);
	}
	for (int i = 0; i < channel; i++)
		delta.push_back(Mat::zeros(row, col, CV_64FC1));
	int index = 0;
	for (int i = 0; i < channel; i++)
		for (int m = 0; m < row; m++)
			for (int n = 0; n < col; n++)
				delta[i].at<double>(m, n) = delta_linear[index++];
	return delta;
}
void Prediction(vector<double> result) {//根据输出向量预测结果
	cout << "\033[36m预测结果：" << endl;
	cout << "[ ";
	double max = result[0];
	int max_index = 0;
	for (int i = 0; i < result.size(); i++) {
		cout << result[i] << " ";
		if (result[i] > max) {
			max_index = i;
			max = result[i];
		}
	}

	cout << "]" << endl;
	cout << "最有可能是：" << max_index << "\033[0m" << endl;
}

Mat Convolution(Mat src, Mat core, int pad_length) {
	//cout << "卷积：" << endl;
	if (src.type() != CV_64FC1 || core.type() != CV_64FC1) {
		cout << "矩阵种类不匹配！" << endl;
	}
	copyMakeBorder(src, src, pad_length, pad_length, pad_length, pad_length, BORDER_CONSTANT, 0);
	Mat dst = Mat::zeros(src.rows - core.rows + 1, src.cols - core.cols + 1, CV_64FC1);
	//cout << "src:" << src.rows << "×" << src.cols << endl;
	//cout << "core:" << core.rows << "×" << core.cols << endl;
	//cout << "dst:" << dst.rows << "×" << dst.cols << endl;
	//cout << "pad_length:" << pad_length << endl;
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			for (int m = 0; m < core.rows; m++) {
				for (int n = 0; n < core.cols; n++) {
					dst.at<double>(i, j) += 
						core.at<double>(m, n) * src.at<double>(i + m, j + n);
				}
			}
		}
	}
	return dst;
}
Mat Convolution2(Mat src, Mat core, int pad_length) {
	//cout << "卷积：" << endl;
	if (src.type() != CV_64FC1 || core.type() != CV_64FC1) {
		cout << "矩阵种类不匹配！" << endl;
	}
	Mat dst;
	copyMakeBorder(src, src, pad_length, pad_length, pad_length, pad_length, BORDER_CONSTANT, 0);
	filter2D(src, dst, -1, core, Point(0, 0), 0, BORDER_CONSTANT);
	//cout << src.rows << " "<<core.rows << endl;
	Rect rect(0, 0, src.rows - core.rows + 1, src.cols - core.cols + 1);
	dst = dst(rect);
	return dst;
}
