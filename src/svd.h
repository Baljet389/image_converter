#ifndef SVD_H
#define SVD_H

#include <vector>
#include <cmath>
#include "bidiagonal.h"
#include "utility.h"

template<typename A>
struct RotationEntry;
template<typename A>
SVD<A> calcSVD(const Matrix<A>& mat);
template<typename A>
A calculateWilkinsonShift(const SubMatrixView<A>& mat);
template<typename A>
struct SVDWorkspace;
template<typename A>
void applyInitialRightGivensRotation(SubMatrixView<A>&              mat,
                                     A                              shift,
                                     std::vector<RotationEntry<A>>& rot,
                                     bool                           updateV);
template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>&              mat,
                             uint32_t                       k,
                             std::vector<RotationEntry<A>>& rot,
                             bool                           updateU);
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>&              mat,
                              uint32_t                       k,
                              std::vector<RotationEntry<A>>& rot,
                              bool                           updateV);
template<typename A>
std::vector<uint32_t>
svdIteration(SubMatrixView<A>& mat, SVDWorkspace<A>& ws, SVD<A>* svd = nullptr);
template<typename A>
void svdRecursive(SubMatrixView<A>& subMat, SVDWorkspace<A>& ws, SVD<A>* svd = nullptr);
template<typename A>
void applyFusedRotation(Matrix<A>&                           target,
                        const std::vector<RotationEntry<A>>& rotations,
                        uint32_t                             offset);


template<typename A>
struct SVDWorkspace {
    std::vector<RotationEntry<A>> rotL;
    std::vector<RotationEntry<A>> rotR;


    void reserve(uint32_t n) {
        rotL.reserve(n);
        rotR.reserve(n);
    }

    void clear() {
        rotL.clear();
        rotR.clear();
    }
};
template<typename A>
struct RotationEntry {
    A        s, c;
    uint32_t k;
    RotationEntry(A S, A C, uint32_t K) :
        s(S),
        c(C),
        k(K) {}
};


template<typename A>
SVD<A> calcSVD(const Matrix<A>& mat) {
    bool     isWide = mat.cols > mat.rows;
    uint32_t r      = mat.rows;
    uint32_t c      = mat.cols;
    if (isWide)
        std::swap(r, c);

    Matrix<A> U(r, r);
    U.setIdentity();

    Matrix<A> V(c, c);
    V.setIdentity();

    Timer  timer;
    SVD<A> svd(U, isWide ? physicalTranspose(mat) : mat, V);
    timer.startTimer();
    calculateBidiagonalForm(svd.S, &svd);
    auto elapsed = timer.stopTimer();
    std::cout << "Bidiagonalization took " << elapsed.count() << " ms" << std::endl;

    SVDWorkspace<A> ws;
    ws.reserve(std::max(r, c));
    SubMatrixView<A> subMat(svd.S, 0, 0, r, c);
    svdRecursive(subMat, ws, &svd);
    auto svdTime = timer.stopTimer() - elapsed;
    std::cout << "SVD took " << svdTime.count() << " ms" << std::endl;
    if (isWide)
    {
        // If we computed SVD of A^T = U S V^T,
        // then the original A = V S^T U^T
        // Transpose S back to its original wide shape
        svd.S = physicalTranspose(svd.S);
        swapMatrices(svd.U, svd.V);
    }

    return svd;
}


template<typename A>
void svdRecursive(SubMatrixView<A>& subMat, SVDWorkspace<A>& ws, SVD<A>* svd) {
    if (subMat.cols <= 1 || subMat.rows <= 1)
        return;

    std::vector<uint32_t> deflationIndices;
    do
    {
        deflationIndices = svdIteration(subMat, ws, svd);
    } while (deflationIndices.empty());

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
            svdRecursive(subBlock, ws, svd);
        }
        start = deflationIndex + 1;
    }
}
template<typename A>
std::vector<uint32_t> svdIteration(SubMatrixView<A>& mat, SVDWorkspace<A>& ws, SVD<A>* svd) {
    uint32_t n     = std::min(mat.rows, mat.cols);
    A        shift = calculateWilkinsonShift(mat);

    bool updateU = (svd != nullptr && svd->U.data.size() > 0);
    bool updateV = (svd != nullptr && svd->V.data.size() > 0);
    ws.clear();
    std::vector<uint32_t> deflationIndices;
    // Introduce the bulge
    applyInitialRightGivensRotation(mat, shift, ws.rotR, updateV);
    // Chase the bulge

    for (uint32_t k = 0; k < n - 1; k++)
    {
        applyLeftGivensRotation(mat, k, ws.rotL, updateU);
        if (k < n - 2)
            applyRightGivensRotation(mat, k, ws.rotR, updateV);
    }
    if (svd != nullptr)
    {
        applyFusedRotation(svd->U, ws.rotL, mat.rowOffset);
        applyFusedRotation(svd->V, ws.rotR, mat.colOffset);
    }
    A tolerance = 2.2e-15 * calculateBidiagonalFrobreniusNorm(mat);

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
                                     A                              shift,
                                     std::vector<RotationEntry<A>>& rot,
                                     bool                           updateV) {
    A x = mat(0, 0) * mat(0, 0) - shift;
    A y = mat(0, 0) * mat(0, 1);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = 0; i < 2; i++)
    {
        A mat_i_zero = mat(i, 0);
        A mat_i_one  = mat(i, 1);

        mat(i, 0) = c * mat_i_zero + s * mat_i_one;
        mat(i, 1) = s * mat_i_zero - c * mat_i_one;
    }
    if (updateV)
        rot.emplace_back(s, c, 0);
}
template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>&              mat,
                             uint32_t                       k,
                             std::vector<RotationEntry<A>>& rot,
                             bool                           updateU) {
    A x = mat(k, k);
    A y = mat(k + 1, k);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    uint32_t minDim = std::min(mat.rows, mat.cols);
    uint32_t end    = std::min(k + 3, minDim);

    for (uint32_t j = k; j < end; j++)
    {
        A mat_kj  = mat(k, j);
        A mat_k1j = mat(k + 1, j);
        // S = G * S
        mat(k, j)     = c * mat_kj + s * mat_k1j;
        mat(k + 1, j) = s * mat_kj - c * mat_k1j;
    }
    if (updateU)
        rot.emplace_back(s, c, k);
}
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>&              mat,
                              uint32_t                       k,
                              std::vector<RotationEntry<A>>& rot,
                              bool                           updateV) {
    A x = mat(k, k + 1);
    A y = mat(k, k + 2);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = k; i < k + 3; i++)
    {
        A mat_ik1 = mat(i, k + 1);
        A mat_ik2 = mat(i, k + 2);

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

    for (const auto& rot : rotations)
    {
        uint32_t col1 = rot.k + offset;
        uint32_t col2 = col1 + 1;

        A c = rot.c;
        A s = rot.s;

        A* col1Pointer = target.getColumnPointer(col1);
        A* col2Pointer = target.getColumnPointer(col2);

        for (uint32_t i = 0; i < target.rows; i++)
        {
            A a            = col1Pointer[i];
            A b            = col2Pointer[i];
            col1Pointer[i] = c * a + s * b;
            col2Pointer[i] = s * a - c * b;
        }
    }
}

#endif