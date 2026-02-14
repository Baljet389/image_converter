#ifndef QR_H
#define QR_H

#include "bidiagonal.h"
#include "matrix.h"
#include "utility.h"
#include <vector>
#include <cmath>
#include <iostream>

template<typename A>
QR<A> calcQRBlocked(const Matrix<A>& mat, bool updateQ);
template<typename A>
struct QRWorkspace;
template<typename A>
struct BlockUpdateWorkspace;
template<typename A>
void generatePanel(QRWorkspace<A>& ws, uint32_t currBlockSize, uint32_t col);
template<typename A>
A calcNormedHouseholder(std::vector<A>& v);
template<typename A>
void updateTMatrix(QRWorkspace<A>& ws, A tauLeft, uint32_t col, uint32_t rowOffset);
template<typename A>
void applyBlockReflectorToR(BlockUpdateWorkspace<A>& blockws);
template<typename A>
void applyBlockReflectorToQ(BlockUpdateWorkspace<A>& blockws);

template<typename A>
struct QRWorkspace {
    Matrix<A>&     R;
    Matrix<A>&     V;
    Matrix<A>&     T;
    std::vector<A> vL;
    QRWorkspace(Matrix<A>& r, Matrix<A>& v, Matrix<A>& t) :
        R(r),
        V(v),
        T(t) {}

    void reserve(uint32_t n) { vL.reserve(n); }
};
template<typename A>
struct BlockUpdateWorkspace {
    Matrix<A>& W;
    Matrix<A>& Y;

    Matrix<A>& W2;
    Matrix<A>& Y2;

    SubMatrixView<A>& activeV;
    SubMatrixView<A>& activeT;
    SubMatrixView<A>& trailingR;
    SubMatrixView<A>& trailingQ;
    BlockUpdateWorkspace(Matrix<A>&        w,
                         Matrix<A>&        y,
                         Matrix<A>&        w2,
                         Matrix<A>&        y2,
                         SubMatrixView<A>& v,
                         SubMatrixView<A>& t,
                         SubMatrixView<A>& r,
                         SubMatrixView<A>& q) :
        W(w),
        Y(y),
        W2(w2),
        Y2(y2),
        activeV(v),
        activeT(t),
        trailingR(r),
        trailingQ(q) {}
};
template<typename A>
QR<A> calcQRBlocked(const Matrix<A>& mat, bool updateQ) {
    Matrix<A> QMatrix(mat.rows, mat.rows);
    QMatrix.setIdentity();
    QR<A> qr(QMatrix, mat);

    Matrix<A>& Q            = qr.Q;
    Matrix<A>& R            = qr.R;
    uint32_t   minDimension = std::min(R.rows, R.cols);
    uint32_t   blockSize    = 64;

    Matrix<A> V(R.rows, blockSize);
    Matrix<A> T(blockSize, blockSize);

    Matrix<A> W(blockSize, R.cols);
    Matrix<A> Y(blockSize, R.cols);

    Matrix<A> W2(R.rows, blockSize);
    Matrix<A> Y2(R.rows, blockSize);

    QRWorkspace<A> ws(R, V, T);
    ws.reserve(R.rows);
    for (uint32_t i = 0; i < minDimension; i += blockSize)
    {
        uint32_t currBlockSize = std::min(blockSize, minDimension - i);
        V.setValue(0);
        generatePanel(ws, currBlockSize, i);


        const uint32_t nextCol       = i + currBlockSize;
        const uint32_t trailingCols  = R.cols - nextCol;
        const uint32_t currentHeight = R.rows - i;

        SubMatrixView<A> activeV(V, i, 0, currentHeight, currBlockSize);
        SubMatrixView<A> activeT(T, 0, 0, currBlockSize, currBlockSize);
        SubMatrixView<A> trailingR(R, i, nextCol, currentHeight, trailingCols);
        SubMatrixView<A> trailingQ(Q, 0, i, Q.rows, currentHeight);

        BlockUpdateWorkspace<A> blockws(W, Y, W2, Y2, activeV, activeT, trailingR, trailingQ);

        // Update R = R - V * T^T * V^T * R

        if (trailingCols > 0)
        {
            applyBlockReflectorToR(blockws);
        }

        // Update Q = Q - Q * V * T * V^T

        if (updateQ)
        {
            applyBlockReflectorToQ(blockws);
        }
    }
    return qr;
}

template<typename A>
void generatePanel(QRWorkspace<A>& ws, uint32_t currBlockSize, uint32_t col) {

    std::vector<A>& vL = ws.vL;
    Matrix<A>&      R  = ws.R;
    Matrix<A>&      V  = ws.V;

    for (uint32_t i = col; i < col + currBlockSize; i++)
    {
        const uint32_t m = R.rows - i;
        vL.resize(m);

        for (uint32_t j = 0; j < m; j++)
            vL[j] = R(i + j, i);

        A tauLeft = calcNormedHouseholder(vL);

        A* VColumnPtr = V.getColumnPointer(i - col) + i;
        for (uint32_t j = 0; j < m; j++)
            VColumnPtr[j] = vL[j];

        for (uint32_t j = i; j < currBlockSize + col; j++)
        {
            A  dot        = 0;
            A* RColumnPtr = R.getColumnPointer(j) + i;
            for (uint32_t k = 0; k < m; k++)
                dot += vL[k] * RColumnPtr[k];
            A prod = tauLeft * dot;
            for (uint32_t k = 0; k < m; k++)
                RColumnPtr[k] -= vL[k] * prod;
        }

        updateTMatrix(ws, tauLeft, i - col, i);
    }
}
template<typename A>
void updateTMatrix(QRWorkspace<A>& ws, A tauLeft, uint32_t col, uint32_t rowOffset) {

    // $$T_j = \begin{pmatrix} T_{j-1} & -\tau_j T_{j-1} V_{j-1}^T v_j \\ 0 & \tau_j \end{pmatrix}$$


    std::vector<A>& vL = ws.vL;
    Matrix<A>&      V  = ws.V;
    Matrix<A>&      T  = ws.T;

    T(col, col) = tauLeft;
    if (col == 0)
        return;

    std::vector<A> temp(col, A(0));

    for (uint32_t j = 0; j < col; j++)
    {
        A dot = 0;
        for (uint32_t k = 0; k < vL.size(); k++)
        {
            dot += vL[k] * V(k + rowOffset, j);
        }
        temp[j] = dot;
    }
    for (uint32_t j = 0; j < col; j++)
    {
        A sum = 0;
        for (uint32_t k = j; k < col; k++)
        {
            sum += T(j, k) * temp[k];
        }
        T(j, col) = -tauLeft * sum;
    }
}
template<typename A>
void applyBlockReflectorToR(BlockUpdateWorkspace<A>& blockws) {
    SubMatrixView<A>& trailingR = blockws.trailingR;
    SubMatrixView<A>& activeV   = blockws.activeV;
    SubMatrixView<A>& activeT   = blockws.activeT;

    const uint32_t currBlockSize = activeT.rows;
    // const uint32_t currentHeight = activeV.rows;
    const uint32_t trailingCols = trailingR.cols;

    SubMatrixView<A> W(blockws.W, 0, 0, currBlockSize, trailingCols);
    SubMatrixView<A> Y(blockws.Y, 0, 0, currBlockSize, trailingCols);

    // Step A: W = V^T * R_trailing
    // Size: blockSize x trailingCols = (currBlockSize x currentHeight) * (currentHeight x trailingCols)

    transposeMultMatrix(activeV, trailingR, W);

    // Step B: Y = T^T * W
    // Size: blockSize x trailingCols = (currBlockSize x currBlockSize) * (currBlockSize x trailingCols)

    transposeMultMatrix(activeT, W, Y);

    // Step C: R_trailing = R_trailing - V * Y
    // Size: (currentHeight x trailingCols) = (currentHeight x currBlockSize) * (currBlockSize x trailingCols)
    matrixMinusMatrixMultMatrix(trailingR, activeV, Y);
}
template<typename A>
void applyBlockReflectorToQ(BlockUpdateWorkspace<A>& blockws) {
    SubMatrixView<A>& trailingQ = blockws.trailingQ;
    SubMatrixView<A>& activeV   = blockws.activeV;
    SubMatrixView<A>& activeT   = blockws.activeT;

    const uint32_t currBlockSize = activeT.rows;
    const uint32_t rows          = trailingQ.rows;
    // const uint32_t cols          = trailingQ.cols;

    SubMatrixView<A> W2(blockws.W2, 0, 0, rows, currBlockSize);
    SubMatrixView<A> Y2(blockws.Y2, 0, 0, rows, currBlockSize);

    // Step A: W = trailingQ * V
    // Size: (rows, cols) * (cols, currBlockSize) = (rows, currBlockSize)

    matrixMultMatrix(trailingQ, activeV, W2);

    // Step B: Y = W * T
    // Size: (rows, currBlockSize) * (currBlockSize, currBlockSize) = (rows, currBlockSize)
    matrixMultMatrix(W2, activeT, Y2);

    // Step C: trailingQ = trailingQ - W_updated * V^T
    // Size: (rows, currBlockSize) * (currBlockSize, cols) = (rows, cols)

    matrixMinusMatrixMultTranspose(trailingQ, Y2, activeV);
}
template<typename A>
A calcNormedHouseholder(std::vector<A>& v) {
    A sigma = A(0);
    for (uint32_t i = 1; i < v.size(); ++i)
        sigma += v[i] * v[i];

    if (sigma == A(0))
    {
        return A(0);
    }

    A x0   = v[0];
    A norm = std::sqrt(x0 * x0 + sigma);

    A alpha = (x0 <= A(0)) ? norm : -norm;

    A tau = (alpha - x0) / alpha;

    A inv = A(1) / (x0 - alpha);
    v[0]  = A(1);
    for (uint32_t i = 1; i < v.size(); ++i)
        v[i] *= inv;

    return tau;
}

#endif