#ifndef SVD_H
#define SVD_H

#include <vector>
#include <cmath>
#include <iostream>
#include "bidiagonal.h"
#include <omp.h>

struct Timer;
template<typename A>
struct RotationEntry;
template<typename A>
SVD<A> calcSVD(Matrix<A>& mat, Timer* t);
template<typename A>
A calculateWilkinsonShift(const SubMatrixView<A>& mat);
template<typename A>
void applyInitialRightGivensRotation(SubMatrixView<A>&              mat,
                                     const A&                       shift,
                                     std::vector<RotationEntry<A>>& rot,
                                     SVD<A>*                        svd);
template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>&              mat,
                             const uint32_t&                k,
                             std::vector<RotationEntry<A>>& rot,
                             SVD<A>*                        svd = nullptr);
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>&              mat,
                              const uint32_t&                k,
                              std::vector<RotationEntry<A>>& rot,
                              SVD<A>*                        svd = nullptr);
template<typename A>
std::vector<uint32_t> svdIteration(SubMatrixView<A>& mat, SVD<A>* svd = nullptr);
template<typename A>
void svdRecursive(SubMatrixView<A>& subMat, SVD<A>* svd = nullptr);
template<typename A>
void applyFusedRotation(Matrix<A>&                           target,
                        const std::vector<RotationEntry<A>>& rotations,
                        uint32_t                             offset);


template<typename A>
struct RotationEntry {
    A        s, c;
    uint32_t k;
    RotationEntry(A S, A C, uint32_t K) :
        s(S),
        c(C),
        k(K) {}
};
struct Timer {
    std::chrono::steady_clock::time_point start;

    void                      startTimer() { start = std::chrono::steady_clock::now(); }
    std::chrono::milliseconds stopTimer() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
};

template<typename A>
SVD<A> calcSVD(Matrix<A>& mat, Timer* t) {
    bool     isWide = mat.cols > mat.rows;
    uint32_t r      = mat.rows;
    uint32_t c      = mat.cols;
    if (isWide)
        std::swap(r, c);

    Matrix<A> U(r, r);
    U.setIdentity();

    Matrix<A> V(c, c);
    V.setIdentity();

    SVD<A> svd(U, isWide ? mat.physicalTranspose() : mat, V);
    calculateBidiagonalForm(svd.S, &svd);
    if (t != nullptr)
    {
        auto timeSVD = t->stopTimer();
        std::cout << "Elapsed: " << timeSVD.count() << " ms\n";
    }


    SubMatrixView<A> subMat(svd.S, 0, 0, r, c);
    svdRecursive(subMat, &svd);

    if (isWide)
    {
        // If we computed SVD of A^T = U S V^T,
        // then the original A = V S^T U^T
        // Transpose S back to its original wide shape
        svd.S = svd.S.physicalTranspose();
        swapMatrices(svd.U, svd.V);
    }

    return svd;
}


template<typename A>
void svdRecursive(SubMatrixView<A>& subMat, SVD<A>* svd) {
    if (subMat.cols <= 1 || subMat.rows <= 1)
        return;

    std::vector<uint32_t> deflationIndices = svdIteration(subMat, svd);

    if (deflationIndices.empty())
    {
        svdRecursive(subMat, svd);
        return;
    }
    uint32_t start = 0;
    deflationIndices.push_back(std::min(subMat.rows, subMat.cols) - 1);
    for (const auto& deflationIndex : deflationIndices)
    {
        uint32_t blockRows = deflationIndex - start + 1;
        uint32_t blockCols = deflationIndex - start + 1;
        if (blockCols > 0 && blockRows > 0)
        {
            SubMatrixView<A> subBlock(subMat.mat, subMat.rowOffset + start,
                                      subMat.colOffset + start, blockRows, blockCols);
            svdRecursive(subBlock, svd);
        }
        start = deflationIndex + 1;
    }
}
template<typename A>
std::vector<uint32_t> svdIteration(SubMatrixView<A>& mat, SVD<A>* svd) {
    uint32_t                      n     = std::min(mat.rows, mat.cols);
    A                             shift = calculateWilkinsonShift(mat);
    std::vector<RotationEntry<A>> rotL;
    std::vector<RotationEntry<A>> rotR;

    rotL.reserve(n - 1);
    rotR.reserve(n - 1);
    // Introduce the bulge
    applyInitialRightGivensRotation(mat, shift, rotR, svd);
    // Chase the bulge

    for (uint32_t k = 0; k < n - 1; k++)
    {
        applyLeftGivensRotation(mat, k, rotL, svd);
        if (k < n - 2)
            applyRightGivensRotation(mat, k, rotR, svd);
    }
    if (svd != nullptr)
    {
        applyFusedRotation(svd->U, rotL, mat.rowOffset);
        applyFusedRotation(svd->V, rotR, mat.colOffset);
    }
    A tolerance = 2.2e-15 * calculateBidiagonalFrobreniusNorm(mat);

    std::vector<uint32_t> deflationIndices;
    for (uint32_t i = 0; i < n - 1; i++)
    {
        A offDiagonalElement = mat.get(i, i + 1);
        if (std::abs(offDiagonalElement) < tolerance)
        {
            mat(i, i + 1) = A(0);
            deflationIndices.push_back(i);
        }
    }
    return deflationIndices;
}

template<typename A>
A calculateWilkinsonShift(const SubMatrixView<A>& mat) {
    uint32_t n = std::min(mat.rows, mat.cols);
    if (n < 2)
        return A(0);

    A am   = mat(n - 1, n - 1) * mat(n - 1, n - 1);
    A am_1 = mat(n - 2, n - 2) * mat(n - 2, n - 2) + mat(n - 2, n - 1) * mat(n - 2, n - 1);
    A bm_1 = mat(n - 2, n - 2) * mat(n - 2, n - 1);

    A delta = (am_1 - am) / 2.0;
    A denom = delta + std::copysign(std::hypot(bm_1, delta), delta);

    if (denom == 0)
        return am;

    return am - (bm_1 * bm_1) / denom;
}

template<typename A>
void applyInitialRightGivensRotation(SubMatrixView<A>&              mat,
                                     const A&                       shift,
                                     std::vector<RotationEntry<A>>& rot,
                                     SVD<A>*                        svd) {
    bool updateV = (svd != nullptr && svd->V.data.size() > 0);
    A    x       = mat(0, 0) * mat(0, 0) - shift;
    A    y       = mat(0, 0) * mat(0, 1);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = 0; i < 2; i++)
    {
        A  mat_i_zero = mat(i, 0);
        A& mat_i_one  = mat(i, 1);

        mat(i, 0) = c * mat_i_zero + s * mat_i_one;
        mat(i, 1) = s * mat_i_zero - c * mat_i_one;
    }
    if (updateV)
        rot.emplace_back(s, c, 0);
}
template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>&              mat,
                             const uint32_t&                k,
                             std::vector<RotationEntry<A>>& rot,
                             SVD<A>*                        svd) {
    bool updateU = (svd != nullptr && svd->U.data.size() > 0);
    A    x       = mat(k, k);
    A    y       = mat(k + 1, k);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    uint32_t minDim = std::min(mat.rows, mat.cols);
    uint32_t end    = std::min(k + 3, minDim);

    for (uint32_t j = k; j < end; j++)
    {
        A  mat_kj  = mat(k, j);
        A& mat_k1j = mat(k + 1, j);
        // S = G * S
        mat(k, j)     = c * mat_kj + s * mat_k1j;
        mat(k + 1, j) = s * mat_kj - c * mat_k1j;
    }
    if (updateU)
        rot.emplace_back(s, c, k);
}
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>&              mat,
                              const uint32_t&                k,
                              std::vector<RotationEntry<A>>& rot,
                              SVD<A>*                        svd) {
    bool updateV = (svd != nullptr && svd->V.data.size() > 0);
    A    x       = mat(k, k + 1);
    A    y       = mat(k, k + 2);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = k; i < k + 3; i++)
    {
        A  mat_ik1 = mat(i, k + 1);
        A& mat_ik2 = mat(i, k + 2);

        mat(i, k + 1) = c * mat_ik1 + s * mat_ik2;
        mat(i, k + 2) = s * mat_ik1 - c * mat_ik2;
    }
    if (updateV)
        rot.emplace_back(s, c, k + 1);
}

template<typename A>
void applyFusedRotation(Matrix<A>&                           target,
                        const std::vector<RotationEntry<A>>& rotations,
                        uint32_t                             offset) {
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for (const auto& rot : rotations)
    {
        uint32_t col1 = rot.k + offset;
        uint32_t col2 = col1 + 1;

        A c = rot.c;
        A s = rot.s;

        A* col1Pointer = &(target.getData()[col1 * target.rows]);
        A* col2Pointer = &(target.getData()[col2 * target.rows]);

        for (int32_t i = 0; i < target.rows; i++)
        {
            A a            = col1Pointer[i];
            A b            = col2Pointer[i];
            col1Pointer[i] = c * a + s * b;
            col2Pointer[i] = s * a - c * b;
        }
    }
}

#endif