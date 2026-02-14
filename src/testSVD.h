#ifndef TEST_SVD_H
#define TEST_SVD_H

#include <cassert>
#include <iostream>
#include <string>
#include "utility.h"
#include "svd.h"
#include "qr.h"

void testAll();
void testSquareMatrixSVD();
void testThinMatrixSVD();
void testWideMatrixSVD();
void testQR();
template<typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat);
template<typename A>
void assertQR(QR<A>& qr, Matrix<A>& origiMat);
template<typename A>
void assertOrthorgonality(const Matrix<A>& mat);


void testAll() {
    testSquareMatrixSVD();
    testThinMatrixSVD();
    testWideMatrixSVD();
    testQR();
}
void testSquareMatrixSVD() {
    Matrix<double> mat(10, 10);
    fillMatrixRandomValues(mat, double(0), double(1));
    SVD<double> svd = calcSVD(mat);
    assertSVD<double>(svd, mat);
}
// m > n
void testThinMatrixSVD() {
    Matrix<double> mat(10, 6);
    fillMatrixRandomValues(mat, double(0), double(1));

    SVD<double> svd = calcSVD(mat);
    assertSVD(svd, mat);
}
// m < n
void testWideMatrixSVD() {
    Matrix<double> mat(6, 10);
    fillMatrixRandomValues(mat, double(0), double(1));


    SVD<double> svd = calcSVD(mat);
    assertSVD(svd, mat);
}
void testQR() {
    Matrix<double> mat(10, 10);
    fillMatrixRandomValues(mat, double(0), double(1));
    Timer t;
    t.startTimer();
    QR<double> qr      = calcQRBlocked(mat, true);
    auto       elapsed = t.stopTimer();
    std::cout << "QR took: " << elapsed.count() << "ms\n";
    assertOrthorgonality(qr.Q);
    assertQR(qr, mat);
}

template<typename A>
void assertSVD(SVD<A>& svd, Matrix<A>& origiMat) {
    Matrix<A>&  mat = svd.S;
    std::string whichMatrix =
      (mat.rows == mat.cols) ? "square" : ((mat.rows > mat.cols) ? "thin" : "wide");

    Matrix<A> finalRecon(mat.rows, mat.cols);
    reconstructSVD(svd, finalRecon, std::min(mat.rows, mat.cols));

    A maxDiff = A(0);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            A diff = std::abs(finalRecon.get(i, j) - origiMat.get(i, j));
            ASSERT(diff < 2.2e-10, "SVD Test failed; diff: " << diff << "\n");
            maxDiff = std::max(maxDiff, diff);
        }
    }
    std::cout << "SVD: Test successful for " << whichMatrix << " matrix! \n";
    std::cout << "Max difference: " << maxDiff << "\n";
}
template<typename A>
void assertQR(QR<A>& qr, Matrix<A>& origiMat) {
    Matrix<A>&  Q           = qr.Q;
    Matrix<A>&  R           = qr.R;
    std::string whichMatrix = (R.rows == R.cols) ? "square" : ((R.rows > R.cols) ? "thin" : "wide");
    Matrix<A>   recon(R.rows, R.cols);
    matrixProduct(Q, R, recon);
    A maxDiff = A(0);
    for (uint32_t i = 0; i < R.rows; i++)
    {
        for (uint32_t j = 0; j < R.cols; j++)
        {
            A diff = std::abs(recon.get(i, j) - origiMat.get(i, j));
            ASSERT(diff < 2.2e-10, "QR Test failed");
            maxDiff = std::max(maxDiff, diff);
        }
    }
    std::cout << "QR: Test successful for " << whichMatrix << " matrix! \n";
    std::cout << "Max difference: " << maxDiff << "\n";
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
                ASSERT(std::abs(dot - A(1)) < 2.2e-10, "Orthogonal Test failed");
            else
                ASSERT(std::abs(dot) < 2.2e-10, "Orthogonal Test failed");
        }
    }
    std::cout << "Orthogonality test successful! \n";
}


#endif