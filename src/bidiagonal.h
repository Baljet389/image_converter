#include <vector>
#include <cmath>
#include "matrix.h"


template <typename A> bool calculateHouseholderVector(std::vector<A>& x);
template <typename A> A calculateTau(const std::vector<A>& v);
template <typename A> void calculateBidiagonalForm(Matrix<A>& mat);
template <typename A> A calculateBidiagonalFrobreniusNorm(const SubMatrixView<A>& subMat);


template <typename A> void calculateBidiagonalForm(Matrix<A>& mat){
    uint32_t minDimension = std::min(mat.rows, mat.cols);
    for(uint32_t i = 0; i < minDimension; i++){

        // left Householder transformation column i
        std::vector<A> xLeft(mat.rows - i);
        for(uint32_t j = i; j < mat.rows; j++) xLeft[j - i] = mat.get(j, i);
        
        // calculate v = x - (sign(x1))*||x||*e1)
        bool reflectorNeededLeft = calculateHouseholderVector(xLeft);
        if(reflectorNeededLeft) {
        // and tau = 2 / (v^T * v)
        A tauLeft = calculateTau(xLeft);
        // apply the transformation to A from the left: A = A - tau*v*(v^T*A)
        // to submatrix A(i:rows, i:cols)
        for(uint32_t j = i; j < mat.cols;j++){
            A dot = 0;
            for(uint32_t k = i; k < mat.rows; k++) dot += xLeft[k - i] * mat.get(k , j);
            for(uint32_t k = i; k < mat.rows; k++) {
                A val = mat.get(k, j) - tauLeft * xLeft[k - i] * dot;
                mat.set(k, j, val);
            }
        }
        }
        if(i >= minDimension - 1) break;
        // right Householder transformation row i
        std::vector<A> xRight(mat.cols - (i + 1));
        for(uint32_t j = i + 1; j < mat.cols; j++) xRight[j - (i + 1)] = mat.get(i, j);

        // calculate v = x - (sign(x1))*||x||*e1)
        bool reflectorNeededRight = calculateHouseholderVector(xRight);
        if(!reflectorNeededRight) continue;
        // and tau = 2 / (v^T * v)
        A tauRight = calculateTau(xRight);
        // apply the transformation to A from the right: A = A - tau*(A*v)*(v^T)
        // to submatrix A(i:rows, i+1:cols)
        for(uint32_t j = i; j < mat.rows; j++){
            A dot = 0;
            for(uint32_t k = i + 1; k < mat.cols; k++) dot += mat.get(j, k) * xRight[k - (i + 1)];
            for(uint32_t k = i + 1; k < mat.cols; k++) {
                A val = mat.get(j, k) - tauRight * dot * xRight[k - (i + 1)];
                mat.set(j, k, val);
            }
        }
    }
}

template <typename A> bool calculateHouseholderVector(std::vector<A>& x){
    A normX = 0.0;
    for(const auto& xi : x) normX += xi * xi;

    normX = std::sqrt(normX);
    if(normX == 0) return false;

    A sign = (x[0] >= 0) ? 1 : -1;
    x[0] += sign * normX;
    return true;
}
template <typename A> A calculateTau(const std::vector<A>& v){
    A tau = 0;
    for(const auto& vi : v) tau += vi * vi;
    tau = 2 / tau;
    return tau;
}

template <typename A> A calculateBidiagonalFrobreniusNorm(const SubMatrixView<A>& subMat){
    A sum = 0;
    uint32_t minDimension = std::min(subMat.subCols, subMat.subRows);
    for(uint32_t i = 0; i < minDimension; i++){
        A diag = subMat.get(i, i);
        sum += diag * diag;
        if(i < minDimension - 1){
            A superDiag = subMat.get(i, i + 1);
            sum += superDiag * superDiag;
        }
    }
    return std::sqrt(sum);
}