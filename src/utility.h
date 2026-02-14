#ifndef UTILITY_H
#define UTILITY_H

#include <random>
#include <chrono>
#include "matrix.h"
#include <cassert>

struct Timer;
template<typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop);
template<typename A>
void reconstructSVD(SVD<A>& svd, Matrix<A>& res, uint32_t rank);
template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void transposeMultMatrix(MatrixInterface<A, DerivedA>& matA,
                         MatrixInterface<A, DerivedB>& matB,
                         MatrixInterface<A, DerivedR>& result);

template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixMultMatrix(MatrixInterface<A, DerivedA>& matA,
                      MatrixInterface<A, DerivedB>& matB,
                      MatrixInterface<A, DerivedR>& result);

// calculates A = A - B * C^T
template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixMinusMatrixMultTranspose(MatrixInterface<A, DerivedA>& matA,
                                    MatrixInterface<A, DerivedB>& matB,
                                    MatrixInterface<A, DerivedR>& matC);

struct Timer {
    std::chrono::steady_clock::time_point start;

    void                      startTimer() { start = std::chrono::steady_clock::now(); }
    std::chrono::milliseconds stopTimer() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
};
template<typename A>
void fillMatrixRandomValues(Matrix<A>& mat, A disStart, A disStop) {
    std::random_device          rd;
    std::mt19937                generator(rd());
    std::normal_distribution<A> distribution(disStart, disStop);
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            mat(i, j) = distribution(generator);
        }
    }
}
template<typename A>
void reconstructSVD(SVD<A>& svd, Matrix<A>& res, uint32_t rank) {
    res.setValue(A(0));

    Matrix<A>& U = svd.U;
    Matrix<A>& S = svd.S;
    Matrix<A>& V = svd.V;

    uint32_t rows = S.getRows();
    uint32_t cols = S.getCols();

    for (uint32_t j = 0; j < cols; j++)
    {
        A* resPtr = res.getColumnPointer(j);
        for (uint32_t k = 0; k < rank; k++)
        {
            A* UColPtr = U.getColumnPointer(k);
            A  scalar  = S(k, k) * V(j, k);
            for (uint32_t i = 0; i < rows; i++)
            {
                resPtr[i] += UColPtr[i] * scalar;
            }
        }
    }
}

template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void transposeMultMatrix(MatrixInterface<A, DerivedA>& matA,
                         MatrixInterface<A, DerivedB>& matB,
                         MatrixInterface<A, DerivedR>& result) {

    assert(matA.isContiguous());
    assert(matB.isContiguous());
    assert(matA.getRows() == matB.getRows());

    const uint32_t colsA = matA.getCols();
    const uint32_t colsB = matB.getCols();
    const uint32_t rowsB = matB.getRows();

    for (uint32_t r = 0; r < colsA; r++)
    {
        A* colAPtr = matA.getColumnPointer(r);
        for (uint32_t k = 0; k < colsB; k++)
        {
            A* colBPtr = matB.getColumnPointer(k);
            A  dot     = 0;
            for (uint32_t c = 0; c < rowsB; c++)
            {
                dot += colAPtr[c] * colBPtr[c];
            }
            result(r, k) = dot;
        }
    }
}
template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixMultMatrix(MatrixInterface<A, DerivedA>& matA,
                      MatrixInterface<A, DerivedB>& matB,
                      MatrixInterface<A, DerivedR>& result) {

    assert(matA.isContiguous());
    assert(matB.isContiguous());
    assert(matA.getCols() == matB.getRows());

    const uint32_t rowsA = matA.getRows();
    const uint32_t colsA = matA.getCols();
    const uint32_t colsB = matB.getCols();

    result.setValue(0);
    for (uint32_t r = 0; r < colsB; r++)
    {
        A* resColPtr = result.getColumnPointer(r);
        for (uint32_t k = 0; k < colsA; k++)
        {
            A  val        = matB(k, r);
            A* matAColPtr = matA.getColumnPointer(k);
            for (uint32_t c = 0; c < rowsA; c++)
            {
                resColPtr[c] += matAColPtr[c] * val;
            }
        }
    }
}

template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixMinusMatrixMultTranspose(MatrixInterface<A, DerivedA>& matA,
                                    MatrixInterface<A, DerivedB>& matB,
                                    MatrixInterface<A, DerivedR>& matC) {

    assert(matA.isContiguous());
    assert(matB.isContiguous());
    assert(matA.getRows() == matB.getRows());
    assert(matA.getCols() == matC.getRows());
    assert(matB.getCols() == matC.getCols());

    const uint32_t rowsA = matA.getRows();
    const uint32_t colsA = matA.getCols();
    const uint32_t colsB = matB.getCols();

    for (uint32_t r = 0; r < colsA; r++)
    {
        A* aColPtr = matA.getColumnPointer(r);
        for (uint32_t k = 0; k < colsB; k++)
        {
            A* bColPtr = matB.getColumnPointer(k);
            A  val     = matC(r, k);
            for (uint32_t c = 0; c < rowsA; c++)
            {
                aColPtr[c] -= bColPtr[c] * val;
            }
        }
    }
}
template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixMinusMatrixMultMatrix(MatrixInterface<A, DerivedA>& matA,
                                 MatrixInterface<A, DerivedB>& matB,
                                 MatrixInterface<A, DerivedR>& matC) {

    // Dimensional Asserts for A = A - (B * C)
    assert(matA.isContiguous());  // Assuming your getColumnPointer implies this
    assert(matB.isContiguous());

    assert(matA.getRows() == matB.getRows());  // Heights must match
    assert(matB.getCols() == matC.getRows());  // Inner dimension (k)
    assert(matA.getCols() == matC.getCols());  // Widths must match

    const uint32_t rowsA  = matA.getRows();
    const uint32_t colsA  = matA.getCols();
    const uint32_t innerK = matB.getCols();

    for (uint32_t j = 0; j < colsA; j++)
    {
        A* aColPtr = matA.getColumnPointer(j);

        for (uint32_t k = 0; k < innerK; k++)
        {
            const A* bColPtr = matB.getColumnPointer(k);
            A        val     = matC(k, j);

            for (uint32_t i = 0; i < rowsA; i++)
            {
                aColPtr[i] -= bColPtr[i] * val;
            }
        }
    }
}
#endif
