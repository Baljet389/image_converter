#ifndef TEST_SVD_H
#define TEST_SVD_H

#include <cassert>
#include <random>
#include <chrono>
#include <iostream>
#include "svd.h"

void testSquareMatrixSVD();
void testThinMatrixSVD();
void testWideMatrixSVD();


void testSquareMatrixSVD() {
    Matrix<double_t>                   mat(1000, 1000);
    std::random_device                 rd;
    std::mt19937                       generator(rd());
    std::normal_distribution<double_t> distribution(0.0, 1.0);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            mat.set(i, j, distribution(generator));
        }
    }
    Timer t;
    std::cout << "Start Timer \n";
    t.startTimer();

    SVD<double_t> svd = calcSVD(mat, &t);

    auto timeSVD = t.stopTimer();
    std::cout << "Elapsed: " << timeSVD.count() << " ms\n";
    // U * S
    Matrix<double_t> recon(mat.rows, mat.cols);
    matrixProduct(svd.U, svd.S, recon);
    // Multiply by V^T
    Matrix<double_t>        finalRecon(mat.rows, mat.cols);
    TransposeView<double_t> vt(svd.V);

    matrixProduct(recon, vt, finalRecon);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            assert(std::abs(finalRecon.get(i, j) - mat.get(i, j)) < 1e-6);
        }
    }
    auto time = t.stopTimer();
    std::cout << "Test successful for rectengular matrix! \n";
    std::cout << "Elapsed: " << time.count() << " ms\n";
}
// m > n
void testThinMatrixSVD() {
    Matrix<double_t>                   mat(10, 6);
    std::random_device                 rd;
    std::mt19937                       generator(rd());
    std::normal_distribution<double_t> distribution(0.0, 1.0);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            mat.set(i, j, distribution(generator));
        }
    }
    SVD<double_t> svd = calcSVD(mat, nullptr);
    // U * S
    Matrix<double_t> recon(mat.rows, mat.cols);
    matrixProduct(svd.U, svd.S, recon);
    // Multiply by V^T
    Matrix<double_t>        finalRecon(mat.rows, mat.cols);
    TransposeView<double_t> vt(svd.V);
    matrixProduct(recon, vt, finalRecon);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            assert(std::abs(finalRecon.get(i, j) - mat.get(i, j)) < 1e-6);
        }
    }
    std::cout << "Test successful for thin matrix! \n";
}
// m < n
void testWideMatrixSVD() {
    Matrix<double_t> mat(6, 10);

    std::random_device                 rd;
    std::mt19937                       generator(rd());
    std::normal_distribution<double_t> distribution(0.0, 1.0);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            mat.set(i, j, distribution(generator));
        }
    }

    SVD<double_t> svd = calcSVD(mat, nullptr);
    // U * S
    Matrix<double_t> recon(mat.rows, mat.cols);
    matrixProduct(svd.U, svd.S, recon);
    // Multiply by V^T
    Matrix<double_t>        finalRecon(mat.rows, mat.cols);
    TransposeView<double_t> vt(svd.V);
    matrixProduct(recon, vt, finalRecon);

    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            double_t diff = std::abs(finalRecon.get(i, j) - mat.get(i, j));
            assert(diff < 1e-6);
        }
    }
    std::cout << "Test successful for wide matrix! \n";
}

#endif