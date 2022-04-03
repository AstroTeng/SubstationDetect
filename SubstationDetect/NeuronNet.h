#pragma once
#include "Matrix.h"
#include "ElementGenerate.h"
#include "CoutShape.h"
#include <random>
#include <fstream>

#define LAST_ROW 16
#define LAST_COL 16
#define LAST_CHAN 20
Matrix_ Softmax(Matrix_ in);
Matrix_ Relu(Matrix_ in);
Matrix_mat Relu(Matrix_mat in);
Matrix_mat Softmax(Matrix_mat in);
class Layer {
public:
	int in_num;
	int out_num;
};
class Layer_fc : public Layer {
public:
	Matrix_ weight;
	Matrix_ grad_w;
	Matrix_ bias;
	Matrix_ grad_b;
	Matrix_ output_z;
	Matrix_ output_a;
	Matrix_ delta_in;
	Matrix_ delta_out;
	Layer_fc(int in, int out) {
		in_num = in;
		out_num = out;
		Matrix_ w(out, in);
		grad_w = weight = w;
		Matrix_ o(out, 1);
		grad_b = delta_in = bias = output_a = output_z = o;
		Matrix_ d(in, 1);
		delta_out = d;
	}
	void Randomize();
#define ACT_RELU 0
#define ACT_SOFTMAX 1
	void Forward(Matrix_ last_active, int act_type);
	void Backward(Matrix_ last_out, int act_type);
	void Grad(Matrix_ last_active);
	void Update(float learning_rate);
};
class Layer_cnn : public Layer {
public:
	int convLength;
	Matrix_mat conv;
	Matrix_mat grad_w;
	Matrix_ bias;
	Matrix_ grad_b;
	Matrix_mat output_z;
	Matrix_mat output_a;
	Matrix_mat delta_in;
	Matrix_mat delta_out;
	Layer_cnn(int in, int out, int conv_length) {
		in_num = in;
		out_num = out;
		convLength = conv_length;
		Matrix_mat w(out, in, conv_length);
		grad_w = conv = w;
		Matrix_mat o(out, 1);
		delta_in = output_a = output_z = o;
		Matrix_mat d(in, 1);
		delta_out = d;
		Matrix_ b(out, 1);
		grad_b=bias = b;
	}
	void Randomize();
	void Forward(Matrix_mat last_active, int act_type);
	void Backward(Matrix_mat last_out, int act_type);
	void Grad(Matrix_mat last_active);
	void Update(float learning_rate);
};




class Layer_pool {
public:
	int chan_num;
	int last_length;
	Matrix_mat output;
	Matrix_mat loc_buff;
	Matrix_mat delta_out;
	Layer_pool(int chan, int l_length) {
		chan_num = chan;
		last_length = l_length;
		Matrix_mat o(chan_num, 1, last_length / 2);
		output = o;
		Matrix_mat l(chan_num, 1, last_length / 2);
		loc_buff = l;
		Matrix_mat d(chan_num, 1,l_length);
		delta_out = d;
	}
	void Forward(Matrix_mat last_active);
	void Backward(Matrix_mat last_out);
};

class Net {
public:
#define L_CNN 0
#define L_POOL 1
#define L_FC 2
	char name[64];
	String target;
	int target_lable;
	Net(String name_,int l_row, int l_col, int l_chan);
	int layer_index;
	int cnn_index;
	int pool_index;
	int fc_index;
	int l_row;
	int l_col;
	int l_chan;
	String *n_buff;
	vector<int> layer_seq;
	vector<Layer_cnn> seq_cnn;
	vector<Layer_pool> seq_pool;
	vector<Layer_fc> seq_fc;
	void Add(Layer_cnn in);
	void Add(Layer_pool in);
	void Add(Layer_fc in);
	void ShowSeq();
	void ShowEnd();
	int ShowResult();
	void Randomize();
	void Forward();
	void Backward();
	void GradUp();
	void Target(String in, bool lable_inable);
	void Update(float rate);
	void SaveData(String path);
	void LoadData(String path);
#define E_OUTZ 0
#define E_OUTA 1
#define E_DELTA_IN 2
#define E_DELTA_OUT 3
#define E_CONV 4
#define E_GRAD 5
	void OutPic(int type, int index, int element,String path);
private:
	void All2Zero();
	void All2Max();
};
class Net2 :public Net {
	void Add(Layer_cnn in);
	void Add(Layer_pool in);
	void Add(Layer_fc in);
};



Matrix_ Flat(Matrix_mat in);
Matrix_mat UnFlat(Matrix_ in, int row, int col, int chan);

Matrix_mat PicLoad(String path,bool if_binar,bool if_blur);
Matrix_mat PicLoad2(String path, bool if_binar);
void PicOut(Matrix_mat in, String path);
void PicOutStack(Matrix_mat in, String path);

