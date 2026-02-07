#ifndef TEST_SVD_H
#define TEST_SVD_H

#include <cassert>
#include <random>
#include <chrono>
#include <iostream>
#include <string>
#include "svd.h"

void testSquareMatrixSVD();
void testThinMatrixSVD();
void testWideMatrixSVD();
struct Timer;
template <typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat);
template <typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop);

struct Timer {
    std::chrono::steady_clock::time_point start;

    void                      startTimer() { start = std::chrono::steady_clock::now(); }
    std::chrono::milliseconds stopTimer() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
};


void testSquareMatrixSVD() {
    Matrix<double_t>                   mat(1000, 1000);
    fillMatrixRandomValues(mat,double_t(0),double_t(1));
    Timer t;
    std::cout << "Start Timer \n";
    t.startTimer();

    SVD<double_t> svd = calcSVD(mat);

    auto timeSVD = t.stopTimer();
    std::cout << "Elapsed: " << timeSVD.count() << " ms\n";
    assertSVD(svd,mat);
	auto time = t.stopTimer();
    std::cout << "Elapsed: " << time.count() << " ms\n";
}
// m > n
void testThinMatrixSVD() {
    Matrix<double_t>                   mat(10, 6);
    fillMatrixRandomValues(mat,double_t(0),double_t(1));

    SVD<double_t> svd = calcSVD(mat);
    assertSVD(svd, mat);
}
// m < n
void testWideMatrixSVD() {
    Matrix<double_t> mat(6, 10);
    fillMatrixRandomValues(mat,double_t(0),double_t(1));


    SVD<double_t> svd = calcSVD(mat);
	assertSVD(svd, mat);
}

template <typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat){
	Matrix<A>& mat = svd.S;
	std::string whichMatrix = (mat.rows == mat.cols) ? "square" :
					((mat.rows > mat.cols) ? "thin" : "wide");
 	
	Matrix<A> recon(mat.rows, mat.cols);
    matrixProduct(svd.U, svd.S, recon);
    // Multiply by V^T
    Matrix<A>        finalRecon(mat.rows, mat.cols);
    TransposeView<A> vt(svd.V);
    matrixProduct(recon, vt, finalRecon);

    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            A diff = std::abs(finalRecon.get(i, j) - origiMat.get(i, j));
            assert(diff < 1e-6);
        }
    }
    std::cout << "Test successful for " << whichMatrix << " matrix! \n";
}
template <typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop){
	std::random_device                 rd;
    std::mt19937                       generator(rd());
    std::normal_distribution<double_t> distribution(disStart, disStop);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            mat(i, j) = distribution(generator);
        }
    }
}

#endif