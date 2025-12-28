#include <vector>
#include <cmath>
#include "bidiagonal.h"

template <typename T> struct SVD;
template <typename A> SVD<A> calcSVD(Matrix<A>& mat);
template <typename A> void claculateSingularValues(Matrix<A>& mat);
template <typename A> A calculateWilkinsonShift(const SubMatrixView<A>& mat);
template <typename A> void applyInitialRightGivensRotation(SubMatrixView<A>& mat, A shift);

template <typename A> void applyLeftGivensRotation(SubMatrixView<A>& mat, uint32_t k);
template <typename A> void applyRightGivensRotation(SubMatrixView<A>& mat, uint32_t k);
template <typename A> std::vector<A> svdIteration(SubMatrixView<A>& mat);
template <typename A> void svdRecursive(SubMatrixView<A>& subMat);

template <typename T> 
struct SVD{
    Matrix<T>& U, S, V;
    SVD(Matrix<T>& u, const Matrix<T>& s, const Matrix<T>& v) : U(u), S(s), V(v) {}

};


template <typename A> SVD<A> 
calcSVD(Matrix<A>& mat){
    Matrix<A> U(mat.rows, mat.rows);
    U.setIdentity();

    Matrix<A> V(mat.cols, mat.cols);
    V.setIdentity();

    SVD<A> svd(U, mat, V);
    calculateBidiagonalForm(mat);
    SubMatrixView<A> subMat(mat, 0, 0, mat.rows, mat.cols);
    svdRecursive(subMat);
}
template <typename A> void claculateSingularValues(Matrix<A>& mat){
    calculateBidiagonalForm(mat);
    SubMatrixView<A> subMat(mat, 0, 0, mat.rows, mat.cols);
    svdRecursive(subMat);
}
template <typename A> void svdRecursive(SubMatrixView<A>& subMat){
    if(subMat.subCols <= 1 || subMat.subRows <= 1) return;

    std::vector<A> deflationIndices = svdIteration(subMat);

    if(deflationIndices.empty()) {
        svdRecursive(subMat);
        return;
    }
    uint32_t start = 0;
    deflationIndices.push_back(std::min(subMat.subRows, subMat.subCols) - 1);
    for(const auto& deflationIndex : deflationIndices){
        uint32_t blockRows = deflationIndex - start + 1;
        uint32_t blockCols = deflationIndex - start + 1;
        if(blockCols > 0 && blockRows > 0){
            SubMatrixView<A> subBlock(subMat.mat, subMat.rowOffset + start, subMat.colOffset + start, blockRows, blockCols);
            svdRecursive(subBlock);
        }
        start = deflationIndex + 1;
    }
}
template <typename A> std::vector<A> svdIteration(SubMatrixView<A>& mat){
    A shift = calculateWilkinsonShift(mat);
    // Introduce the bulge
    applyInitialRightGivensRotation(mat, shift);

    // Chase the bulge
    uint32_t n = std::min(mat.subRows, mat.subCols);
    for(uint32_t k = 0; k < n - 1; k++){
        applyLeftGivensRotation(mat, k);
        if (k < n - 2) applyRightGivensRotation(mat, k);
    }
    A tolerance = 2.2e-15 * calculateBidiagonalFrobreniusNorm(mat);
    
    std::vector<A> deflationIndices;
    for(uint32_t i = 0; i < n - 1; i++){
        A offDiagonalElement = mat.get(i, i + 1);
        if(std::abs(offDiagonalElement) < tolerance){
            mat.set(i, i + 1, 0.0);
            deflationIndices.push_back(i);
        }
    }
    return deflationIndices;
}

template <typename A> A calculateWilkinsonShift(const SubMatrixView<A>& mat){
    uint32_t n = std::min(mat.subRows, mat.subCols);
    if (n < 2) return A(0);

    A am = mat.get(n - 1, n - 1) * mat.get(n - 1, n - 1);
    A am_1 = mat.get(n - 2, n - 2) * mat.get(n - 2, n - 2) + mat.get(n - 2, n - 1) * mat.get(n - 2, n - 1);
    A bm_1 = mat.get(n - 2, n - 2) * mat.get(n - 2,n - 1);
    A delta = (am_1 - am) / 2.0;
    A mu = am - (bm_1 * bm_1) / (delta + std::copysign(delta, std::hypot(delta, bm_1)));
    return mu;
}

template <typename A> void applyInitialRightGivensRotation(SubMatrixView<A>& mat, A shift){
    A x = mat.get(0,0) * mat.get(0,0) - shift;
    A y = mat.get(0,0) * mat.get(0,1);

    A r = std::hypot(x, y);
    if (r == A(0)) return;

    A s = y / r;
    A c = x / r;

    for(uint32_t i = 0; i < mat.subRows; i++){
        A mat_i_zero = mat.get(i,0);
        A mat_i_one = mat.get(i,1);

        mat.set(i, 0, c * mat_i_zero + s * mat_i_one);
        mat.set(i, 1, s * mat_i_zero - c * mat_i_one);
    }
 
}
template <typename A> void applyLeftGivensRotation(SubMatrixView<A>& mat, uint32_t k){
    A x = mat.get(k, k);
    A y = mat.get(k + 1, k);
    
    A r = std::hypot(x, y);
    if (r == A(0)) return;

    A s = y / r;
    A c = x / r;

    for(uint32_t j = 0; j < mat.subCols; j++){
        A mat_zero_j = mat.get(k,j);
        A mat_one_j = mat.get(k+1,j);

        mat.set(k, j, c * mat_zero_j + s * mat_one_j);
        mat.set(k + 1, j, s * mat_zero_j - c * mat_one_j);
    }

}
template <typename A> void applyRightGivensRotation(SubMatrixView<A>& mat, uint32_t k){
    A x = mat.get(k, k + 1);
    A y = mat.get(k, k + 2);
    
    A r = std::hypot(x, y);
    if (r == A(0)) return;

    A s = y / r;
    A c = x / r;

    for(uint32_t i = 0; i < mat.subRows; i++){
        A mat_i_zero = mat.get(i,k +1);
        A mat_i_one = mat.get(i,k+2);

        mat.set(i, k + 1, c * mat_i_zero + s * mat_i_one);
        mat.set(i, k + 2, s * mat_i_zero - c * mat_i_one);
    }

}

