#include <vector>
#include <cmath>
#include <iostream>
#include "bidiagonal.h"

template<typename A>
SVD<A> calcSVD(Matrix<A>& mat);
template<typename A>
void claculateSingularValues(Matrix<A>& mat);
template<typename A>
SVD<A> calculateSVDWithKSingularValues(Matrix<A>& mat, uint32_t k);
template<typename A>
A calculateWilkinsonShift(const SubMatrixView<A>& mat);
template<typename A>
void applyInitialRightGivensRotation(SubMatrixView<A>& mat, A shift, SVD<A>* svd);

template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>& mat, uint32_t k, SVD<A>* svd = nullptr);
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>& mat, uint32_t k, SVD<A>* svd = nullptr);
template<typename A>
std::vector<uint32_t> svdIteration(SubMatrixView<A>& mat, SVD<A>* svd = nullptr);
template<typename A>
void svdRecursive(SubMatrixView<A>& subMat, SVD<A>* svd = nullptr);


template<typename A>
SVD<A> calcSVD(Matrix<A>& mat) {
    bool isWide = mat.cols > mat.rows;

    if (isWide)
        mat.transpose();

    Matrix<A> U(mat.rows, mat.rows);
    U.setIdentity();

    Matrix<A> V(mat.cols, mat.cols);
    V.setIdentity();

    SVD<A> svd(U, mat, V);
    calculateBidiagonalForm(svd.S, &svd);


    SubMatrixView<A> subMat(svd.S, 0, 0, mat.rows, mat.cols);
    svdRecursive(subMat, &svd);

    if (isWide)
    {
        // If we computed SVD of A^T = U S V^T,
        // then the original A = V S^T U^T
        // Transpose S back to its original wide shape
        svd.S.transpose();
        swapMatrices(svd.U, svd.V);
        mat.transpose();
    }

    return svd;
}

// Here the input matrix gets modified
template<typename A>
void claculateSingularValues(Matrix<A>& mat) {
    bool isWide = mat.cols > mat.rows;

    if (isWide)
        mat.transpose();

    calculateBidiagonalForm(mat, static_cast<SVD<A>*>(nullptr));
    SubMatrixView<A> subMat(mat, 0, 0, mat.rows, mat.cols);
    svdRecursive(subMat, static_cast<SVD<A>*>(nullptr));

    if (isWide)
        mat.transpose();
}
template<typename A>
SVD<A> calculateSVDWithKSingularValues(Matrix<A>& mat, uint32_t k) {
    bool isWide = mat.cols > mat.rows;

    if (isWide)
        mat.transpose();

    Matrix<A> U(mat.rows, mat.rows);
    U.setIdentity();

    Matrix<A> V(mat.cols, mat.cols);
    V.setIdentity();

    SVD<A> svd(U, mat, V);
    calculateBidiagonalForm(svd.S, &svd);
    uint32_t         cols = std::min(k, mat.cols);
    uint32_t         rows = std::min(k, mat.rows);
    SubMatrixView<A> subMat(svd.S, 0, 0, rows, cols);

    svdRecursive(subMat, &svd);

    if (isWide)
    {
        svd.S.transpose();
        swapMatrices(svd.U, svd.V);
        mat.transpose();
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
    A shift = calculateWilkinsonShift(mat);
    // Introduce the bulge
    applyInitialRightGivensRotation(mat, shift, svd);
    // Chase the bulge
    uint32_t n = std::min(mat.rows, mat.cols);
    for (uint32_t k = 0; k < n - 1; k++)
    {
        applyLeftGivensRotation(mat, k, svd);
        if (k < n - 2)
            applyRightGivensRotation(mat, k, svd);
    }
    A tolerance = 2.2e-15 * calculateBidiagonalFrobreniusNorm(mat);

    std::vector<uint32_t> deflationIndices;
    for (uint32_t i = 0; i < n - 1; i++)
    {
        A offDiagonalElement = mat.get(i, i + 1);
        if (std::abs(offDiagonalElement) < tolerance)
        {
            mat.set(i, i + 1, 0.0);
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

    A am = mat.get(n - 1, n - 1) * mat.get(n - 1, n - 1);
    A am_1 =
      mat.get(n - 2, n - 2) * mat.get(n - 2, n - 2) + mat.get(n - 2, n - 1) * mat.get(n - 2, n - 1);
    A bm_1 = mat.get(n - 2, n - 2) * mat.get(n - 2, n - 1);

    A delta = (am_1 - am) / 2.0;
    A denom = delta + std::copysign(std::hypot(bm_1, delta), delta);

    if (denom == 0)
        return am;

    return am - (bm_1 * bm_1) / denom;
}

template<typename A>
void applyInitialRightGivensRotation(SubMatrixView<A>& mat, A shift, SVD<A>* svd) {
    bool updateV = (svd != nullptr && svd->V.data.size() > 0);
    A    x       = mat.get(0, 0) * mat.get(0, 0) - shift;
    A    y       = mat.get(0, 0) * mat.get(0, 1);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = 0; i < mat.rows; i++)
    {
        A mat_i_zero = mat.get(i, 0);
        A mat_i_one  = mat.get(i, 1);

        mat.set(i, 0, c * mat_i_zero + s * mat_i_one);
        mat.set(i, 1, s * mat_i_zero - c * mat_i_one);
    }
    if (updateV)
    {
        for (uint32_t i = 0; i < svd->V.rows; i++)
        {
            // update V matrix: V = V * Gr
            A v_i_zero = svd->V.get(i, mat.colOffset + 0);
            A v_i_one  = svd->V.get(i, mat.colOffset + 1);

            svd->V.set(i, mat.colOffset + 0, c * v_i_zero + s * v_i_one);
            svd->V.set(i, mat.colOffset + 1, s * v_i_zero - c * v_i_one);
        }
    }
}
template<typename A>
void applyLeftGivensRotation(SubMatrixView<A>& mat, uint32_t k, SVD<A>* svd) {
    bool updateU = (svd != nullptr && svd->U.data.size() > 0);
    A    x       = mat.get(k, k);
    A    y       = mat.get(k + 1, k);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t j = 0; j < mat.cols; j++)
    {
        A mat_kj  = mat.get(k, j);
        A mat_k1j = mat.get(k + 1, j);
        // S = G * S
        mat.set(k, j, c * mat_kj + s * mat_k1j);
        mat.set(k + 1, j, s * mat_kj - c * mat_k1j);
    }
    if (updateU)
    {
        for (uint32_t i = 0; i < svd->U.rows; i++)
        {
            A u_ik  = svd->U.get(i, mat.rowOffset + k);
            A u_ik1 = svd->U.get(i, mat.rowOffset + k + 1);

            // U = U * G
            svd->U.set(i, mat.rowOffset + k, c * u_ik + s * u_ik1);
            svd->U.set(i, mat.rowOffset + k + 1, s * u_ik - c * u_ik1);
        }
    }
}
template<typename A>
void applyRightGivensRotation(SubMatrixView<A>& mat, uint32_t k, SVD<A>* svd) {
    bool updateV = (svd != nullptr && svd->V.data.size() > 0);
    A    x       = mat.get(k, k + 1);
    A    y       = mat.get(k, k + 2);

    A r = std::hypot(x, y);
    if (r == A(0))
        return;

    A s = y / r;
    A c = x / r;

    for (uint32_t i = 0; i < mat.rows; i++)
    {
        A mat_ik1 = mat.get(i, k + 1);
        A mat_ik2 = mat.get(i, k + 2);

        mat.set(i, k + 1, c * mat_ik1 + s * mat_ik2);
        mat.set(i, k + 2, s * mat_ik1 - c * mat_ik2);
    }
    if (updateV)
    {
        for (uint32_t i = 0; i < svd->V.rows; i++)
        {
            // update V matrix: V = V * Gr
            A v_i_zero = svd->V.get(i, mat.colOffset + k + 1);
            A v_i_one  = svd->V.get(i, mat.colOffset + k + 2);

            svd->V.set(i, mat.colOffset + k + 1, c * v_i_zero + s * v_i_one);
            svd->V.set(i, mat.colOffset + k + 2, s * v_i_zero - c * v_i_one);
        }
    }
}
