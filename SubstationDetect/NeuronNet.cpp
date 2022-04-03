#include "NeuronNet.h"
Net::Net(String name_, int l_row,int l_col,int l_chan) {
	strcpy_s(name, name_.c_str());
	this->l_row = l_row;
	this->l_col = l_col;
	this->l_chan = l_chan;
}
void Net::Add(Layer_cnn in) {
	seq_cnn.push_back(in);
	layer_seq.push_back(L_CNN);
}
void Net::Add(Layer_pool in) {
	seq_pool.push_back(in);
	layer_seq.push_back(L_POOL);
}
void Net::Add(Layer_fc in) {
	seq_fc.push_back(in);
	layer_seq.push_back(L_FC);
}
void Net::ShowSeq() {
	for (int i = 0; i < layer_seq.size(); i++)
		cout << layer_seq[i] << " , ";
	cout << endl;
}
void Net::ShowEnd() {

	cout <<endl<< "展示网络最后一层输出：" << endl;
	switch (layer_seq.back()) {
	case L_CNN: seq_cnn.back().output_a.Show();break;
	case L_POOL: seq_pool.back().output.Show(); break;
	case L_FC:
		seq_fc.back().output_a.Show(); 
		
		break;
	}
	cout << endl;
}
int Net::ShowResult() {
	cout << "预测结果：" << endl;
	int max_index = 0;
	for (int i = 0; i < seq_fc.back().output_a.rows; i++) {
		cout << i << "：" << seq_fc.back().output_a.element[i][0]  << endl;
		if (seq_fc.back().output_a.element[i][0] > seq_fc.back().output_a.element[max_index][0]) {
			max_index = i;
		}
	}
	return max_index;
}
void Net::Randomize() {
	All2Zero();
	while (layer_index < layer_seq.size()) {
		switch (layer_seq[layer_index]) {
		case L_CNN:
			seq_cnn[cnn_index++].Randomize();
			break;
		case L_POOL:
			break;
		case L_FC:
			seq_fc[fc_index++].Randomize();
			break;
		}
		layer_index++;
	}
	cout << "网络权重随机化完成" << endl;
}
void Net::Forward() {
	Matrix_mat in;
	in = PicLoad(target,true,false);
	All2Zero();
	while (layer_index < layer_seq.size()) {
		if (layer_index == 0) {
			
			seq_cnn[cnn_index++].Forward(in,ACT_RELU);
		}
		else {
			Matrix_mat in_mat;
			Matrix_ in_;
			switch (layer_seq[layer_index]) {
			case L_CNN:
				switch (layer_seq[layer_index - 1]) {
				case L_CNN:in_mat = seq_cnn[cnn_index - 1].output_a;break;
				case L_POOL:in_mat = seq_pool[pool_index - 1].output;break;
				}
				seq_cnn[cnn_index++].Forward(in_mat,ACT_RELU);
				break;
			case L_POOL:
				switch (layer_seq[layer_index - 1]) {
				case L_CNN:
					in_mat = seq_cnn[cnn_index - 1].output_a;
					break;
				case L_POOL:
					in_mat = seq_pool[pool_index - 1].output;
					break;
				}
				seq_pool[pool_index++].Forward(in_mat);
				break;
			case L_FC:
				switch (layer_seq[layer_index - 1]) {
				case L_CNN:
					in_ = Flat(seq_cnn[cnn_index - 1].output_a);
					break;
				case L_POOL:
					in_ = Flat(seq_pool[pool_index - 1].output);
					break;
				case L_FC:
					in_ = seq_fc[fc_index - 1].output_a;
					break;
				}
				if (layer_index == layer_seq.size() - 1)
					seq_fc[fc_index++].Forward(in_, ACT_SOFTMAX);
				else
					seq_fc[fc_index++].Forward(in_, ACT_RELU);
				break;
			}
		}
		layer_index++;
	}
	C_YELLOW;
	cout << "网络" << name << "前向传播结束" << endl;
	C_NONE;
}
void Net::Backward() {
	All2Max();
	while (layer_index >= 0) {
		if (layer_index == layer_seq.size() - 1) {
			//int lable;
			//for (int i = 0; i < TOTAL_ELEMENT; i++) {
			//	if (nameBuffer[i] == LabelFetch(target)) {
			//		lable = i;
			//		break;
			//	}
			//	if (i == TOTAL_ELEMENT - 1) {
			//		cout << "未知的图片标签！" << endl;
			//		return;
			//	}
			//}
			Matrix_ tmp ;
			tmp =  seq_fc.back().output_a;
			tmp.element[target_lable][0] -= 1;
			seq_fc[fc_index--].Backward(tmp, ACT_SOFTMAX);
		}
		else {
			Matrix_mat in_mat;
			Matrix_ in_;
			switch (layer_seq[layer_index]) {
			case L_CNN:
				switch (layer_seq[layer_index + 1]) {
				case L_CNN:
					in_mat = seq_cnn[cnn_index + 1].delta_out;
					break;
				case L_POOL:
					in_mat = seq_pool[pool_index + 1].delta_out;
					break;
				case L_FC:
					in_mat = UnFlat(seq_fc[fc_index + 1].delta_out,l_row, l_col, l_chan);
					break;
				}
				seq_cnn[cnn_index--].Backward(in_mat,ACT_RELU);
				break;
			case L_POOL:
				switch (layer_seq[layer_index + 1]) {
				case L_CNN:
					in_mat = seq_cnn[cnn_index + 1].delta_out;
					break;
				case L_POOL:
					in_mat = seq_pool[pool_index + 1].delta_out;
					break;
				case L_FC:
					in_mat = UnFlat(seq_fc[fc_index + 1].delta_out, l_row, l_col, l_chan);
					break;
				}
				seq_pool[pool_index--].Backward(in_mat);
				break;
			case L_FC:
				switch (layer_seq[layer_index + 1]) {
				case L_FC:
					in_ = seq_fc[fc_index + 1].delta_out;
					break;
				}
				seq_fc[fc_index--].Backward(in_, ACT_RELU);
			}
		}
		layer_index--;
	}
	cout << "网络" << name << "反向传播结束" << endl;
}
void Net::GradUp() {
	All2Zero();
	while (layer_index < layer_seq.size()) {
		if (layer_index == 0) {
			Matrix_mat in = PicLoad(target, true, false);
			seq_cnn[cnn_index++].Grad(in);
		}
		else {
			Matrix_mat in_mat;
			Matrix_ in_;
			switch (layer_seq[layer_index]) {
			case L_CNN:
				switch (layer_seq[layer_index - 1]) {
				case L_CNN: in_mat = seq_cnn[cnn_index-1].output_a;break;
				case L_POOL: in_mat = seq_pool[pool_index-1].output; break;
					break;
				}
				seq_cnn[cnn_index++].Grad(in_mat);
				break;
			case L_POOL:pool_index++; break;
			case L_FC:
				switch (layer_seq[layer_index - 1]) {
				case L_CNN:in_ = Flat(seq_cnn[cnn_index - 1].output_a);break;
				case L_POOL:in_ = Flat(seq_pool[pool_index - 1].output); break;
				case L_FC:in_ = seq_fc[fc_index - 1].output_a;break;
				}
				seq_fc[fc_index++].Grad(in_);
				break;
			}
		}
		layer_index++;
	}
	cout << "网络" << name << "权重计算完成" << endl;
}
void Net::Target(String in,bool lable_inable) {
	target = in;
	if (lable_inable) {
		for (int i = 0; i < seq_fc.back().output_a.rows ; i++) {
			if (to_string(i) == LabelFetch(target)) {
				target_lable = i;
				break;
			}
			if (i == seq_fc.back().output_a.rows - 1) {
				cout << "未知的图片标签！将设为0" << endl;
				target_lable = 0;
				break;
			}
		}
	}
}
void Net::Update(float rate) {
	for (int index = 0; index < seq_cnn.size(); index++)
		seq_cnn[index].Update(rate);
	for (int index = 0; index < seq_fc.size(); index++)
		seq_fc[index].Update(rate);
	cout << "网络" << name << "权重更新完成" << endl;
}
void Net::SaveData(String path) {
	ofstream out;
	cout << "网络"<<name<<"保存数据至" << path << " ...";
	out.open(path, ios::binary);
	//保存名字用于读取时验证
	out.write((char*)&name, 64);
	//以下保存CNN层的权重数据
	for (int index = 0; index < seq_cnn.size(); index++) 
		for (int i = 0; i < seq_cnn[index].conv.rows; i++) {
			for (int j = 0; j < seq_cnn[index].conv.cols; j++)
				for (int m = 0; m < seq_cnn[index].conv.element[i][j].rows; m++)
					for (int n = 0; n < seq_cnn[index].conv.element[i][j].cols; n++)
						out.write((char*)&seq_cnn[index].conv.element[i][j].at<float>(m, n), sizeof(float));
			out.write((char*)&seq_cnn[index].bias.element[i][0], sizeof(float));
		}
	//以下保存FC层的权重数据
	for(int index=0;index<seq_fc.size();index++)
		for (int i = 0; i < seq_fc[index].weight.rows; i++) {
			for (int j = 0; j < seq_fc[index].weight.cols; j++)
				out.write((char*)&seq_fc[index].weight.element[i][j], sizeof(float));
			out.write((char*)&seq_fc[index].bias.element[i][0], sizeof(float));
		}
	out.close();
	cout << "完成！" << endl;
}
void Net::LoadData(String path) {
	ifstream in;
	cout << "网络" << name << "读取数据从" << path << " ...";
	in.open(path, ios::binary);
	char tmpname[64];
	in.read((char*)&tmpname, 64);
	//使用函数strcmp(s1, s2)

	//	当s<s2 返回负值
	//	当s = s2 返回0
	//	当s>s2返回正值
	if (strcmp(tmpname, name)==0)
		cout << "匹配成功！" << endl;
	else {
		cout << "匹配失败！" << endl;
		in.close();
		return;
	}
	//以下读取CNN层的权重数据
	for (int index = 0; index < seq_cnn.size(); index++)
		for (int i = 0; i < seq_cnn[index].conv.rows; i++) {
			for (int j = 0; j < seq_cnn[index].conv.cols; j++)
				for (int m = 0; m < seq_cnn[index].conv.element[i][j].rows; m++)
					for (int n = 0; n < seq_cnn[index].conv.element[i][j].cols; n++)
						in.read((char*)&seq_cnn[index].conv.element[i][j].at<float>(m, n), sizeof(float));
			in.read((char*)&seq_cnn[index].bias.element[i][0], sizeof(float));
		}
	//以下读取FC层的权重数据
	for (int index = 0; index < seq_fc.size(); index++)
		for (int i = 0; i < seq_fc[index].weight.rows; i++) {
			for (int j = 0; j < seq_fc[index].weight.cols; j++)
				in.read((char*)&seq_fc[index].weight.element[i][j], sizeof(float));
			in.read((char*)&seq_fc[index].bias.element[i][0], sizeof(float));
		}
	in.close();
	cout << "读取完成！" << endl;
}
void Net::OutPic(int type, int index, int element,String path) {
	Matrix_mat tmp;
	String add_path;
	add_path = String(name) + "_";
	switch (type) {
	default:
	case L_CNN:
		if (index >= seq_cnn.size()) {
			cout << "编号超出范围" << endl;
			return;
		}
		add_path = "cnn" + to_string(index) + "_";
		switch (element) {
		case E_OUTZ:tmp = seq_cnn[index].output_z;			add_path += "output_z";	 break;
		case E_OUTA:tmp = seq_cnn[index].output_a; add_path += "output_a"; break;
		case E_DELTA_IN:tmp = seq_cnn[index].delta_in; add_path += "delta_in"; break;
		case E_DELTA_OUT:tmp = seq_cnn[index].delta_out; add_path += "delta_out"; break;
		case E_CONV:tmp = seq_cnn[index].conv; add_path += "conv"; break;
		case E_GRAD:tmp = seq_cnn[index].grad_w; add_path += "grad"; break;
		}
		break;
	case L_POOL:
		if (index >= seq_pool.size()) {
			cout << "编号超出范围" << endl;
			return;
		}
		add_path = "pool" + to_string(index) + "_";
		switch (element) {
		case E_OUTZ:
		case E_OUTA:tmp = seq_pool[index].output; 
			add_path += "output";	
			break;
		case E_DELTA_IN:
		case E_DELTA_OUT:tmp = seq_pool[index].delta_out; 
			add_path += "delta";
			break;
		case E_CONV: break;
		case E_GRAD: break;
		}
		break;
	}
	PicOutStack(tmp, path + add_path);
}
void Net::All2Zero() {
	layer_index = 0;
	cnn_index = 0;
	pool_index = 0;
	fc_index = 0;
}
void Net::All2Max() {
	layer_index = layer_seq.size() - 1;
	cnn_index = seq_cnn.size() - 1;
	pool_index = seq_pool.size() - 1;
	fc_index = seq_fc.size() - 1;
}

Matrix_ Relu(Matrix_ in) {
	Matrix_ tmp ;
	tmp = in;
	for (int i = 0; i < in.rows; i++)
		for (int j = 0; j < in.cols; j++)
			if (tmp.element[i][j] < 0)
				tmp.element[i][j] = 0;
	return tmp;
}
Matrix_ Softmax(Matrix_ in) {
	Matrix_ tmp;
	tmp = in;
	float sum = 0;
	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++) {
			tmp.element[i][j] = exp(tmp.element[i][j]);
			sum += tmp.element[i][j];
		}
			
	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++)
			tmp.element[i][j] /= sum;
	return tmp;
}

void Layer_fc::Randomize() {
	default_random_engine e{ (unsigned int)time(NULL) };
	float std = 2.0 / out_num;
	normal_distribution<float> d(0, std);//正态分布初始化，左边为均值，右边为方差
	for (int i = 0; i < out_num; i++) {
		for (int j = 0; j < in_num; j++)
			weight.element[i][j] = d(e);
		bias.element[i][0] = 0;
	}
	return;
}
void Layer_fc::Forward(Matrix_ last_active,int act_type) {
	output_z = weight * last_active + bias;
	switch (act_type) {
	default:
	case ACT_RELU:
		output_a = Relu(output_z);
		break;
	case ACT_SOFTMAX:
		output_a = Softmax(output_z);
		break;
	}

}
void Layer_fc::Backward(Matrix_ last_out, int act_type) {
	switch (act_type) {
	default:
	case ACT_RELU:
		for (int i = 0; i < output_z.rows; i++)
			if (output_z.element[i][0] < 0)
				last_out.element[i][0] = 0;
		break;
	case ACT_SOFTMAX:
		break;
	}
	delta_in = last_out;
	delta_out = Transpose(weight) * delta_in;
}
void Layer_fc::Grad(Matrix_ last_active) {
	grad_w = grad_w + delta_in * Transpose(last_active);
	grad_b = grad_b + delta_in;
}
void Layer_fc::Update(float learning_rate) {
	weight = weight + (grad_w * (learning_rate * -1));
	bias = bias + (grad_b * (learning_rate * -1));
	Matrix_ tmpGradW(weight.rows, weight.cols);
	grad_w = tmpGradW;
	Matrix_ tmpGradB(bias.rows, bias.cols);
	grad_b = tmpGradB;
}

Matrix_mat Relu(Matrix_mat in) {
	Matrix_mat tmp ;
	tmp = in;
	//tmp.Show();
	for (int i = 0; i < in.rows; i++)
		for (int j = 0; j < in.cols; j++)
			for (int m = 0; m < in.element[i][j].rows; m++)
				for (int n = 0; n < in.element[i][j].cols; n++)
					if (tmp.element[i][j].at<float>(m, n) < 0)
						tmp.element[i][j].at<float>(m, n) = 0;
	return tmp;
}
Matrix_mat Softmax(Matrix_mat in) {
	Matrix_mat tmp;
	tmp = in;
	for (int i = 0; i < in.rows; i++)
		for (int j = 0; j < in.cols; j++) {
			long double sum = 0;
			for (int m = 0; m < in.element[i][j].rows; m++)
				for (int n = 0; n < in.element[i][j].cols; n++) {
					tmp.element[i][j].at<float>(m, n) = exp(tmp.element[i][j].at<float>(m, n));
					sum += tmp.element[i][j].at<float>(m, n);
				}
			for (int m = 0; m < in.element[i][j].rows; m++)
				for (int n = 0; n < in.element[i][j].cols; n++)
					tmp.element[i][j].at<float>(m, n) /= sum;
		}
	return tmp;
}

void Layer_cnn::Randomize() {
	default_random_engine e{ (unsigned int)time(NULL) };
	float std = 2.0 / out_num;
	normal_distribution<float> d(0, std);//正态分布初始化，左边为均值，右边为方差
	for (int i = 0; i < out_num; i++) {
		for (int j = 0; j < in_num; j++)
			for (int m = 0; m < conv.element[i][j].rows; m++)
				for (int n = 0; n < conv.element[i][j].cols; n++)
					conv.element[i][j].at<float>(m, n) = d(e);
		bias.element[i][0] = 0;
	}
	return;
}
void Layer_cnn::Forward(Matrix_mat last_active, int act_type) {
	output_z = conv.Mult_CV(last_active, false);
	for (int i = 0; i < output_z.rows; i++)
		for (int j = 0; j < output_z.cols; j++)
			output_z.element[i][j] += bias.element[i][0];
	switch (act_type) {
	case ACT_RELU:
		output_a = Relu(output_z);
		break;
	case ACT_SOFTMAX:
		output_a = Softmax(output_z);
		break;
	}

}
void Layer_cnn::Backward(Matrix_mat last_out,int act_type) {
	Matrix_mat tmp ;
	tmp = last_out;
	switch (act_type) {
	case ACT_RELU:
		for (int i = 0; i < output_z.rows; i++)
			for (int m = 0; m < output_z.element[i][0].rows; m++)
				for (int n = 0; n < output_z.element[i][0].cols; n++) {
					//cout << "[" << output_z.element[i][0].at<float>(m, n) << "]" << endl;
					if (output_z.element[i][0].at<float>(m, n) < 0)
						tmp.element[i][0].at<float>(m, n) = 0;
				}
		break;
	case ACT_SOFTMAX:
		break;
	}
	delta_in = tmp;
	delta_out = Transpose(conv).Mult_CV(last_out, true);
}
void Layer_cnn::Grad(Matrix_mat last_active) {
	grad_w = grad_w + delta_in.Mult_CLASSIC(Transpose(last_active), this->convLength / 2);
	grad_b = grad_b + Shrink(delta_in);
}
void Layer_cnn::Update(float learning_rate) {
	conv = conv + (grad_w * (learning_rate * -1));
	bias = bias + (grad_b * (learning_rate * -1));
	Matrix_mat tmpGradW(conv.rows, conv.cols, convLength);
	grad_w = tmpGradW;
	Matrix_ tmpGradB(bias.rows, bias.cols);
	grad_b = tmpGradB;
}


void Layer_pool::Forward(Matrix_mat last_active) {
	Matrix_mat out_(last_active.rows, last_active.cols, last_active.element[0][0].rows / 2);
	for (int i = 0; i < chan_num; i++) {
		for (int m = 0; m < last_active.element[i][0].rows; m = m + 2)
			for (int n = 0; n < last_active.element[i][0].cols; n = n + 2) {
				float max = last_active.element[i][0].at<float>(m, n);
				//cout << "1:" << max ;
				loc_buff.element[i][0].at<float>(m / 2, n / 2) = 0;
				if (last_active.element[i][0].at<float>(m, n+1) > max) {
					max = last_active.element[i][0].at<float>(m, n + 1);
					loc_buff.element[i][0].at<float>(m / 2, n / 2) = 1;
				}
				if (last_active.element[i][0].at<float>(m+1, n) > max) {
					max = last_active.element[i][0].at<float>(m + 1, n);
					loc_buff.element[i][0].at<float>(m / 2, n / 2) = 2;
				}
				if (last_active.element[i][0].at<float>(m+1, n+1) > max) {
					max = last_active.element[i][0].at<float>(m + 1, n + 1);
					loc_buff.element[i][0].at<float>(m / 2, n / 2) = 3;
				}
				out_.element[i][0].at<float>(m / 2, n / 2) = max;
				//cout << "  2:" << max ;
				//cout << "  3:" << output.element[i][0].at<float>(m / 2, n / 2) << endl;
			}
	}
	this->output = out_;
}
void Layer_pool::Backward(Matrix_mat last_out) {
	Matrix_mat out_(last_out.rows, last_out.cols, last_out.element[0][0].rows * 2);
	for (int i = 0; i < chan_num; i++) {
		for (int m = 0; m < last_out.element[i][0].rows; m ++)
			for (int n = 0; n < last_out.element[i][0].cols; n ++) {
				int o = loc_buff.element[i][0].at<float>(m, n);
				float result = last_out.element[i][0].at<float>(m, n);
				switch (o) {
				case 0:
					out_.element[i][0].at<float>(m * 2, n * 2) = result;
					break;
				case 1:
					out_.element[i][0].at<float>(m * 2, n * 2 + 1) = result;
					break;
				case 2:
					out_.element[i][0].at<float>(m * 2 + 1, n * 2) = result;
					break;
				case 3:
					out_.element[i][0].at<float>(m * 2 + 1, n * 2 + 1) = result;
					break;
				}	
			}
	}
	this->delta_out = out_;
}

Matrix_ Flat(Matrix_mat in) {
	long int total = in.rows * in.element[0][0].rows * in.element[0][0].cols;
	Matrix_ out(total, 1);
	int index = 0;
	for (int i = 0; i < in.cols; i++)
		for (int m = 0; m < in.element[i][0].rows; m++)
			for (int n = 0; n < in.element[i][0].cols; n++) 
				out.element[index++][0] = in.element[i][0].at<float>(m, n);
	return out;
}
Matrix_mat UnFlat(Matrix_ in,int row,int col,int chan) {
	Matrix_mat out(chan, 0, row);
	int index = 0;
	for (int i = 0; i < out.rows; i++)
		for (int m = 0; m < out.element[i][0].rows; m++)
			for (int n = 0; n < out.element[i][0].cols; n++)
				out.element[i][0].at<float>(m, n) = in.element[index++][0];
	return out;
}

Matrix_mat PicLoad(String path,bool if_binar,bool if_blur) {
	Matrix_mat out(1, 1);
	out.element[0][0] = imread(path);
	if(if_binar)ImgBinar(out.element[0][0]);
	if (if_blur)blur(out.element[0][0], out.element[0][0], Size(3, 3));
	bitwise_not(out.element[0][0], out.element[0][0]);
	if (out.element[0][0].type() != CV_32FC1) {
			cvtColor(out.element[0][0], out.element[0][0], COLOR_RGB2GRAY);//将输入的所有转换为灰度图像
			out.element[0][0].convertTo(out.element[0][0], CV_32FC1);
			out.element[0][0] /= 255.0;//归一化
		//cout << "已完成灰度化" << endl;
	}
	resize(out.element[0][0], out.element[0][0], Size(64, 64));
	return out;
}
Matrix_mat PicLoad2(String path, bool if_binar) {
	Matrix_mat out(1, 1);
	out.element[0][0] = imread(path);
	if (if_binar)ImgBinar(out.element[0][0]);

	bitwise_not(out.element[0][0], out.element[0][0]);
	if (out.element[0][0].type() != CV_32FC1) {
		cvtColor(out.element[0][0], out.element[0][0], COLOR_RGB2GRAY);//将输入的所有转换为灰度图像
		out.element[0][0].convertTo(out.element[0][0], CV_32FC1);
		out.element[0][0] /= 255.0;//归一化
	//cout << "已完成灰度化" << endl;
	}
	return out;
}
void PicOut(Matrix_mat in, String path) {
	for(int i=0;i<in.rows;i++)
		for (int j = 0; j < in.cols; j++) {
			Mat show;
			normalize(in.element[i][j], show, 0, 255, NORM_MINMAX);
			imwrite(path + "_[" + to_string(i) + "][" + to_string(j) + "].jpg", show);
		}

}
void PicOutStack(Matrix_mat in, String path) {
	int gap = 1;
	int height = in.rows * in.element[0][0].rows;
	int width = in.cols * in.element[0][0].cols;
	Mat result = Mat(height + gap * (in.rows - 1), width + gap * (in.cols - 1), in.element[0][0].type(), Scalar::all(255));
	for(int i=0;i<in.rows;i++)
		for (int j = 0; j < in.cols; j++) {
			Mat temp_region = result(Rect(
				j * (gap + in.element[0][0].cols),
				i * (gap + in.element[0][0].rows),
				in.element[0][0].cols,
				in.element[0][0].rows
			));
			Mat show;
			normalize(in.element[i][j], show, 0, 255, NORM_MINMAX);
			show.copyTo(temp_region);
		}
	imwrite(path + "_all.jpg", result);

}
