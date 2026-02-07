#ifndef BIDIAGONAL_H
#define BIDIAGONAL_H

#include <vector>
#include <cmath>
#include <cassert>
#include "matrix.h"
#include <iostream>

template<typename A>
bool calculateHouseholderVector(std::vector<A>& x);
template<typename A>
A calculateTau(const std::vector<A>& v);
template<typename A>
void calculateBidiagonalForm(Matrix<A>& mat, SVD<A>* svd);
template<typename A>
A    calculateBidiagonalFrobreniusNorm(const SubMatrixView<A>& subMat);
template<typename A>
struct BidiagonalWorkspace;

template<typename A>
bool applyLeftHouseholder(BidiagonalWorkspace<A> &ws, Matrix<A>& mat, uint32_t i);
template<typename A>
bool applyRightHouseholder(BidiagonalWorkspace<A>& ws, Matrix<A>& mat, uint32_t i);

template<typename A>
void updateU(BidiagonalWorkspace<A>& ws, Matrix<A>& U, uint32_t i);
template<typename A>
void updateV(BidiagonalWorkspace<A>& ws, Matrix<A>& V, uint32_t i);

template<typename A>
struct BidiagonalWorkspace {
    std::vector<A> vL;
    std::vector<A> vR;
    A              tauLeft;
    A              tauRight;
    void reserve(uint32_t n) {
        vL.reserve(n);
        vR.reserve(n);
    }
};


template<typename A>
void calculateBidiagonalForm(Matrix<A>& mat, SVD<A>* svd) {
    bool shouldUpdateU = (svd != nullptr && svd->U.data.size() > 0);
    bool shouldUpdateV = (svd != nullptr && svd->V.data.size() > 0);

    uint32_t minDimension = std::min(mat.rows, mat.cols);
    BidiagonalWorkspace<A> ws;
    ws.reserve(minDimension);
    for (uint32_t i = 0; i < minDimension; i++)
    {
        bool reflectorNeededLeft = applyLeftHouseholder(ws, mat, i);
            
        if (shouldUpdateU && reflectorNeededLeft)
        {
            updateU(ws, svd->U, i);
        }

        if (i == mat.cols - 1)
            break;
        
        bool reflectorNeededRight = applyRightHouseholder(ws, mat, i);
        if (shouldUpdateV && reflectorNeededRight)
        {
            updateV(ws, svd->V, i);
        }
    }
}
template<typename A>
bool applyLeftHouseholder(BidiagonalWorkspace<A>& ws, Matrix<A>& mat, uint32_t i) {
	auto& vL = ws.vL;
	
	const uint32_t m = mat.rows - i;
    const uint32_t n = mat.cols - i;
	vL.resize(m);
    for (uint32_t j = 0; j < m; j++)
		vL[j] = mat(j + i,i);
    // calculate v = x - (sign(x1))*||x||*e1)
    bool reflectorNeededLeft = calculateHouseholderVector(vL);
    if (!reflectorNeededLeft)
        return false;
    
    // and tau = 2 / (v^T * v)
    A tauLeft = calculateTau(vL);
    ws.tauLeft = tauLeft;
    // apply the transformation to A from the left: A = A - tau*v*(v^T*A)
    // to submatrix A(i:rows, i:cols)
    for (uint32_t j = 0; j < n; j++)
    {
        A dot = 0;
        A* matColumnPtr = mat.getColumnPointer(j + i);

        for (uint32_t k = 0; k < m; k++)
            dot += vL[k] * matColumnPtr[k + i];
        A prod = tauLeft * dot;
        for (uint32_t k = 0; k < m; k++)
        {
            matColumnPtr[k + i] -= vL[k] * prod;
        }
    }
    return true;
}
template<typename A>
bool applyRightHouseholder(BidiagonalWorkspace<A>& ws, Matrix<A>& mat, uint32_t i) {

    auto& vR = ws.vR;
    const uint32_t startColumn = i + 1;
    const uint32_t m = mat.rows - i;
    const uint32_t n = mat.cols - startColumn;

    // --- build vR ---
    vR.resize(n);
    for (uint32_t k = 0; k < n; ++k)
        vR[k] = mat(i, startColumn + k);

    // --- compute reflector ---
    if (!calculateHouseholderVector(vR))
        return false;

    const A tau = calculateTau(vR);
    ws.tauRight = tau;

    // --- dots = A_sub * vR ---
    std::vector<A> dots(m, A(0));

    for (uint32_t k = 0; k < n; ++k)
    {
        const A vk = vR[k];
        A* col = mat.getColumnPointer(startColumn + k) + i;

        for (uint32_t j = 0; j < m; ++j)
            dots[j] += col[j] * vk;
    }

    // --- A_sub -= tau * dots * vRᵀ ---
    for (uint32_t k = 0; k < n; ++k)
    {
        const A scale = tau * vR[k];
        A* col = mat.getColumnPointer(startColumn + k) + i;

        for (uint32_t j = 0; j < m; ++j)
            col[j] -= scale * dots[j];
    }

    return true;
}
template<typename A>
void updateU(BidiagonalWorkspace<A>& ws, Matrix<A>& U, uint32_t i) {
    const auto& vL = ws.vL;
	const A tau = ws.tauLeft;

	const uint32_t m = U.rows;
	const uint32_t n = U.cols - i;

	std::vector<A> dots(m, A(0));

	// 1) dot = U * vL   
	for (uint32_t k = 0; k < n; k++)
	{
		const A vk = vL[k];
		A* col = U.getColumnPointer(i + k);   

		for (uint32_t j = 0; j < m; ++j)
			dots[j] += col[j] * vk;
	}

	// 2) U -= tau * dot * vLᵀ 
	for (uint32_t k = 0; k < n; k++)
	{
		const A scale = tau * vL[k];
		A* col = U.getColumnPointer(i + k);

		for (uint32_t j = 0; j < m; j++)
			col[j] -= scale * dots[j];
	}
}
template<typename A>
void updateV(BidiagonalWorkspace<A>& ws, Matrix<A>& V, uint32_t i) {
	const uint32_t startColumn = i + 1;
    const auto& vR = ws.vR;
    const A tau = ws.tauRight;

    const uint32_t m = V.rows;
    const uint32_t n = V.cols - startColumn;

    std::vector<A> dots(m, A(0));

    // 1) dots = V * vR  
    for (uint32_t k = 0; k < n; k++)
    {
        const A vk = vR[k];
        A* col = V.getColumnPointer(startColumn + k);

        for (uint32_t j = 0; j < m; j++)
            dots[j] += col[j] * vk;
    }

    // 2) V -= tau * dots * vRᵀ
    for (uint32_t k = 0; k < n; k++)
    {
        const A scale = tau * vR[k];
        A* col = V.getColumnPointer(startColumn + k);

        for (uint32_t j = 0; j < m; j++)
            col[j] -= scale * dots[j];
    }
}
template<typename A>
bool calculateHouseholderVector(std::vector<A>& x) {
    A normX = 0.0;
    for (const auto& xi : x)
        normX += xi * xi;

    normX = std::sqrt(normX);
    if (normX == 0)
        return false;

    A sign = (x[0] >= 0) ? 1 : -1;
    x[0] += sign * normX;
    return true;
}
template<typename A>
A calculateTau(const std::vector<A>& v) {
    A tau = 0;
    for (const auto& vi : v)
        tau += vi * vi;
    tau = 2 / tau;
    return tau;
}

template<typename A>
A calculateBidiagonalFrobreniusNorm(const SubMatrixView<A>& subMat) {
    A        sum          = 0;
    uint32_t minDimension = std::min(subMat.rows, subMat.cols);
    for (uint32_t i = 0; i < minDimension; i++)
    {
        A diag = subMat.get(i, i);
        sum += diag * diag;
        if (i < minDimension - 1)
        {
            A superDiag = subMat.get(i, i + 1);
            sum += superDiag * superDiag;
        }
    }
    return std::sqrt(sum);
}

#endif