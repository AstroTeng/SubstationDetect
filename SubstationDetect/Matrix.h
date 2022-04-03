
#pragma once
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;
class Matrix {
public:
	int rows;
	int cols;
};
class Matrix_ :public Matrix {
public:
	vector<vector<float>> element;


	Matrix_();
	Matrix_(int rows_, int cols_);
	void Show();
	Matrix_ operator * (Matrix_ mat_r) {
		if (this->cols != mat_r.rows) {
			cout << "无效相乘，将返回左矩阵！" << endl;
			return *(this);
		}
		Matrix_ output(this->rows, mat_r.cols);
		for (int i = 0; i < output.rows; i++)
			for (int j = 0; j < output.cols; j++)
				for (int k = 0; k < this->cols; k++)
					output.element[i][j] += this->element[i][k] * mat_r.element[k][j];
		return output;
	}
	Matrix_ operator + (Matrix_ mat_r) {
		if (this->cols != mat_r.cols) {
			if (this->rows != mat_r.rows)
				cout << "无效相加，将返回左矩阵！" << endl;
			return *(this);
		}
		for (int i = 0; i < mat_r.rows; i++)
			for (int j = 0; j < mat_r.cols; j++)
				mat_r.element[i][j] += this->element[i][j];
		return mat_r;
	}
};
Matrix_ operator + (double num, Matrix_ matrix);
Matrix_ operator + (Matrix_ matrix, double num);
Matrix_ operator - (Matrix_ matrix, double num);
Matrix_ operator * (double num, Matrix_ matrix);
Matrix_ operator * (Matrix_ matrix, double num);
Matrix_ operator /  (Matrix_ matrix, double num);

Matrix_ Transpose(Matrix_ in);



class Matrix_mat :public Matrix {
public:
	vector<vector<Mat>> element;

	Matrix_mat();
	Matrix_mat(int rows_, int cols_);
	Matrix_mat(int rows_, int cols_,int length_);
	Matrix_mat Mult_CV(Matrix_mat mat_r, bool conv180_enable);
	Matrix_mat Mult_CLASSIC(Matrix_mat mat_r, int pad_length);
	void Show();
	Matrix_mat operator = (Matrix_mat mat_r) {
		//cout << "执行赋值：Matrix_mat" << endl;
		vector<vector<Mat>> another;
		this->rows = mat_r.rows;
		this->cols = mat_r.cols;
		for (int i = 0; i < this->rows; i++) {
			vector<Mat> a_row;
			for (int j = 0; j < this->cols; j++)
				a_row.push_back(mat_r.element[i][j].clone());
			another.push_back(a_row);
		}
		this->element = another;
		return *this;
	}
	Matrix_mat operator + (Matrix_mat mat_r) {
		Matrix_mat tmp;
		tmp = mat_r;
		for (int i = 0; i < tmp.rows; i++)
			for (int j = 0; j < tmp.cols; j++)
				tmp.element[i][j] += this->element[i][j];
		return tmp;
	}
	Matrix_mat operator - (Matrix_mat mat_r) {
		Matrix_mat tmp;
		tmp = *this;
		for (int i = 0; i < tmp.rows; i++)
			for (int j = 0; j < tmp.cols; j++)
				tmp.element[i][j] -= mat_r.element[i][j];
		return tmp;
	}
	Matrix_mat operator * (float x) {
		Matrix_mat tmp;
		tmp = *this;
		for (int i = 0; i < tmp.rows; i++)
			for (int j = 0; j < tmp.cols; j++)
				tmp.element[i][j] *= x;
		return tmp;
	}
};

Matrix_mat Transpose(Matrix_mat in);

Mat ConvClassic(Mat src, Mat core, int pad_length);

Matrix_ Shrink(Matrix_mat in);
