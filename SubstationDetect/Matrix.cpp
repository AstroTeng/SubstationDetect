#include "Matrix.h"
Matrix_::Matrix_() {
	rows = 1;
	cols = 1;
	vector<float> a_row;
	a_row.push_back(0);
	element.push_back(a_row);
}
Matrix_::Matrix_(int rows_, int cols_) {
	if (rows_ <= 0) rows_ = 1;
	if (cols_ <= 0) cols_ = 1;
	rows = rows_;
	cols = cols_;
	for (int i = 0; i < rows; i++) {
		vector<float> a_row;
		for (int j = 0; j < cols; j++) {
			a_row.push_back(0);
		}
		element.push_back(a_row);
	}
}
void Matrix_::Show() {
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			cout << element[i][j] << " , ";
			if (j == cols - 1) cout << endl;
		}
}

Matrix_ operator + (double num, Matrix_ matrix) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] +=  num;
	return matrix;
}
Matrix_ operator + (Matrix_ matrix, double num) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] += num;
	return matrix;
}
Matrix_ operator - (Matrix_ matrix, double num) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] -= num;
	return matrix;
}
Matrix_ operator * (double num, Matrix_ matrix) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] *= num;
	return matrix;
}
Matrix_ operator * ( Matrix_ matrix, double num) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] *= num;
	return matrix;
}
Matrix_ operator / (Matrix_ matrix, double num) {
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			matrix.element[i][j] /= num;
	return matrix;
}

Matrix_ Transpose(Matrix_ in) {
	Matrix_ out(in.cols, in.rows);
	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
			out.element[i][j] = in.element[j][i];
	return out;
}
//----------------------------------以上为double矩阵
//----------------------------------以下为Mat矩阵
Matrix_mat::Matrix_mat() {
	rows = 1;
	cols = 1;
	vector<Mat> a_row;
	a_row.push_back(Mat::zeros(1, 1, CV_32FC1));
	element.push_back(a_row);
}
Matrix_mat::Matrix_mat(int rows_, int cols_) {
	if (rows_ <= 0) rows_ = 1;
	if (cols_ <= 0) cols_ = 1;
	rows = rows_;
	cols = cols_;
	for (int i = 0; i < rows; i++) {
		vector<Mat> a_row;
		for (int j = 0; j < cols; j++) {
			a_row.push_back(Mat::zeros(1, 1, CV_32FC1));
		}
		element.push_back(a_row);
	}
}
Matrix_mat::Matrix_mat(int rows_, int cols_,int length_) {
	if (rows_ <= 0) rows_ = 1;
	if (cols_ <= 0) cols_ = 1;
	rows = rows_;
	cols = cols_;
	for (int i = 0; i < rows; i++) {
		vector<Mat> a_row;
		for (int j = 0; j < cols; j++) {
			a_row.push_back(Mat::zeros(length_, length_, CV_32FC1));
		}
		element.push_back(a_row);
	}
}


Matrix_mat Matrix_mat::Mult_CV(Matrix_mat mat_r, bool conv180_enable) {
	if (this->cols != mat_r.rows) {
		cout << "无效相乘，将返回左矩阵！" << endl;
		return *(this);
	}
	Matrix_mat output(this->rows, mat_r.cols);
		for (int i = 0; i < output.rows; i++)
			for (int j = 0; j < output.cols; j++) {
				Mat sum;
				for (int k = 0; k < this->cols; k++) {
					Mat conv = this->element[i][k].clone();
					if (conv180_enable) {
						flip(conv, conv, 0);//卷积矩阵翻转180°
						flip(conv, conv, 1);
					}
					Mat a_result;
					filter2D(mat_r.element[k][j], a_result, -1, conv, Point(-1, -1), 0, BORDER_CONSTANT);
					if (k == 0) sum = a_result.clone();
					else sum += a_result.clone();
					//cout << sum << endl;
				}
				output.element[i][j] = sum.clone();
			}
	return output;
}
Matrix_mat Matrix_mat::Mult_CLASSIC(Matrix_mat mat_r, int pad_length) {
	if (this->cols != mat_r.rows) {
		cout << "无效相乘，将返回左矩阵！" << endl;
		return *(this);
	}
	Matrix_mat output(this->rows, mat_r.cols);

		for (int i = 0; i < output.rows; i++)
			for (int j = 0; j < output.cols; j++) {
				Mat sum;
				for (int k = 0; k < this->cols; k++) {
					Mat conv = this->element[i][k].clone();
					if (k == 0)
						sum = ConvClassic(mat_r.element[k][j], conv, pad_length);
					else {
						Mat a_result;
						a_result = ConvClassic(mat_r.element[k][j], conv, pad_length);
						sum += a_result;
					}
				}
				output.element[i][j] = sum.clone();
			}

		
	
	return output;
}
void Matrix_mat::Show() {
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			cout << "element[" << i << "][" << j << "]:" << endl;
			cout << element[i][j] << endl;
		}
	cout << endl;
}

Matrix_mat Transpose(Matrix_mat in) {
	Matrix_mat out(in.cols, in.rows);
	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
			out.element[i][j] = in.element[j][i].clone();
	return out;
}


Mat ConvClassic(Mat src_, Mat core, int pad_length) {
	Mat dst;
	Mat src;
	src = src_.clone();
	copyMakeBorder(src, src, pad_length, pad_length, pad_length, pad_length, BORDER_CONSTANT, 0);
	filter2D(src, dst, -1, core, Point(0, 0), 0, BORDER_CONSTANT);
	Rect rect(0, 0, src.rows - core.rows + 1, src.cols - core.cols + 1);
	dst = dst(rect);
	return dst;
}

Matrix_ Shrink(Matrix_mat in) {
	Matrix_ tmp(in.rows, in.cols);
	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++)
			tmp.element[i][j] = cv::sum(in.element[i][j])[0];
	return tmp;
}