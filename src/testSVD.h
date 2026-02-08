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
void testQR();
struct Timer;
template<typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat);
template<typename A>
void assertQR(QR<A>& qr, Matrix<A>& origiMat);
template<typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop);
template<typename A>
void assertOrthorgonality(const Matrix<A>& mat);

struct Timer {
    std::chrono::steady_clock::time_point start;

    void                      startTimer() { start = std::chrono::steady_clock::now(); }
    std::chrono::milliseconds stopTimer() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
};


void testSquareMatrixSVD() {
    Matrix<double_t> mat(100, 100);
    fillMatrixRandomValues(mat, double_t(0), double_t(1));
    Timer t;
    std::cout << "Start Timer \n";
    t.startTimer();

    SVD<double_t> svd = calcSVD(mat);

    auto timeSVD = t.stopTimer();
    std::cout << "Elapsed: " << timeSVD.count() << " ms\n";
    assertSVD(svd, mat);
    auto time = t.stopTimer();
    std::cout << "Elapsed: " << time.count() << " ms\n";
}
// m > n
void testThinMatrixSVD() {
    Matrix<double_t> mat(10, 6);
    fillMatrixRandomValues(mat, double_t(0), double_t(1));

    SVD<double_t> svd = calcSVD(mat);
    assertSVD(svd, mat);
}
// m < n
void testWideMatrixSVD() {
    Matrix<double_t> mat(6, 10);
    fillMatrixRandomValues(mat, double_t(0), double_t(1));


    SVD<double_t> svd = calcSVD(mat);
    assertSVD(svd, mat);
}
void testQR() {
    Matrix<double_t> mat(1000, 1000);
    fillMatrixRandomValues(mat, double_t(0), double_t(1));
    Timer t;
    std::cout << "Start Timer \n";
    t.startTimer();
    QR<double_t> qr     = calcQRBlocked(mat, true);
    auto         timeQR = t.stopTimer();
    std::cout << "Elapsed: " << timeQR.count() << " ms\n";
    assertOrthorgonality(qr.Q);
    assertQR(qr, mat);
    auto time = t.stopTimer();
}

template<typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat) {
    Matrix<A>&  mat = svd.S;
    std::string whichMatrix =
      (mat.rows == mat.cols) ? "square" : ((mat.rows > mat.cols) ? "thin" : "wide");

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
template<typename A>
void assertQR(QR<A>& qr, Matrix<A>& origiMat) {
    Matrix<A>&  Q           = qr.Q;
    Matrix<A>&  R           = qr.R;
    std::string whichMatrix = (R.rows == R.cols) ? "square" : ((R.rows > R.cols) ? "thin" : "wide");
    Matrix<A>   recon(R.rows, R.cols);
    matrixProduct(Q, R, recon);
    for (uint32_t i = 0; i < R.rows; i++)
    {
        for (uint32_t j = 0; j < R.cols; j++)
        {
            A diff = std::abs(recon.get(i, j) - origiMat.get(i, j));
            assert(diff < 1e-6);
        }
    }
    std::cout << "Test successful for " << whichMatrix << " matrix! \n";
}
template<typename A>
void assertOrthorgonality(const Matrix<A>& mat) {
    for (uint32_t i = 0; i < mat.cols; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            A dot = 0;
            for (uint32_t k = 0; k < mat.rows; k++)
            {
                dot += mat.get(k, i) * mat.get(k, j);
            }
            if (i == j)
                assert(std::abs(dot - A(1)) < 1e-6);
            else
                assert(std::abs(dot) < 1e-6);
        }
    }
    std::cout << "Orthogonality test successful! \n";
}
template<typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop) {
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