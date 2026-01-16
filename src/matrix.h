#ifndef MATRIX_H
#define MATRIX_H

#include "stdint.h"
#include <vector>
#include <iostream>
#include <omp.h>


template<typename T>
struct Matrix;
template<typename T>
struct TransposeView;
template<typename A>
struct SubMatrixView;
template<typename T, typename Derived>
struct MatrixInterface;
template<typename T>
struct SVD;
template<typename T>
void printMatrixMatlab(Matrix<T>& mat);

template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixProduct(const MatrixInterface<A, DerivedA>& matA,
                   const MatrixInterface<A, DerivedB>& matB,
                   MatrixInterface<A, DerivedR>&       result);

template<typename A, typename B>
void swapMatrices(Matrix<A>& mat1, Matrix<A>& mat2);

template<typename T>
struct Matrix: public MatrixInterface<T, Matrix<T>> {
    std::vector<T> data;
    uint32_t       rows, cols;

    Matrix(uint32_t r, uint32_t c) :
        data(r * c, T(0)),
        rows(r),
        cols(c) {}
    inline void set(uint32_t r, uint32_t c, T value) { data[c * rows + r] = value; }

    inline T  get(uint32_t r, uint32_t c) const { return data[c * rows + r]; }
    inline T& get(uint32_t r, uint32_t c) { return data[c * rows + r]; }

    void setValues(std::initializer_list<T> list) {
        std::copy(list.begin(), list.end(), data.begin());
    }

    void setIdentity() {
        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                set(i, j, (i == j) ? T(1) : T(0));
            }
        }
    }
    T& operator()(uint32_t r, uint32_t c) { return get(r, c); }

    const T         operator()(uint32_t r, uint32_t c) const { return get(r, c); }
    std::vector<T>& getData() { return data; }
    Matrix          physicalTranspose() const {
        Matrix         result(cols, rows);
        const uint32_t tile_size = 64;

        for (uint32_t r = 0; r < rows; r += tile_size)
        {
            for (uint32_t c = 0; c < cols; c += tile_size)
            {
                for (uint32_t i = r; i < std::min(r + tile_size, rows); ++i)
                {
                    for (uint32_t j = c; j < std::min(c + tile_size, cols); ++j)
                    {
                        result(j, i) = (*this)(i, j);
                    }
                }
            }
        }
        return result;
    }
};
template<typename T>
struct TransposeView: public MatrixInterface<T, TransposeView<T>> {
    Matrix<T>& mat;
    uint32_t   rows, cols;

    TransposeView(Matrix<T>& m) :
        mat(m),
        rows(m.cols),
        cols(m.rows) {}

    inline T  get(uint32_t r, uint32_t c) const { return mat.get(c, r); }
    inline T& get(uint32_t r, uint32_t c) { return mat.get(c, r); }

    inline void set(uint32_t r, uint32_t c, T value) { mat.set(c, r, value); }

    std::vector<T>& getData() { return mat.data; }
};
template<typename T>
struct SubMatrixView: public MatrixInterface<T, SubMatrixView<T>> {
    Matrix<T>& mat;
    uint32_t   rowOffset, colOffset;
    uint32_t   rows, cols;

    SubMatrixView(Matrix<T>& m, uint32_t rOff, uint32_t cOff, uint32_t rSize, uint32_t cSize) :
        mat(m),
        rowOffset(rOff),
        colOffset(cOff),
        rows(rSize),
        cols(cSize) {}

    inline T get(uint32_t r, uint32_t c) const { return mat.get(r + rowOffset, c + colOffset); }

    inline T& get(uint32_t r, uint32_t c) { return mat.get(r + rowOffset, c + colOffset); }

    inline void set(uint32_t r, uint32_t c, T value) {
        mat.set(r + rowOffset, c + colOffset, value);
    }

    T& operator()(uint32_t r, uint32_t c) { return get(r, c); }

    const T         operator()(uint32_t r, uint32_t c) const { return get(r, c); }
    std::vector<T>& getData() { return mat.data; }
};
// CRTP
template<typename T, typename Derived>
struct MatrixInterface {
    MatrixInterface() {}

    inline uint32_t getRows() const { return static_cast<const Derived*>(this)->rows; }
    inline uint32_t getCols() const { return static_cast<const Derived*>(this)->cols; }

    inline void set(uint32_t r, uint32_t c, T value) {
        static_cast<Derived*>(this)->set(r, c, value);
    }

    inline T& get(uint32_t r, uint32_t c) { return static_cast<Derived*>(this)->get(r, c); }
    inline T  get(uint32_t r, uint32_t c) const {
        return static_cast<const Derived*>(this)->get(r, c);
    }

    T& operator()(uint32_t r, uint32_t c) { return static_cast<Derived*>(this)->get(r, c); }

    const T operator()(uint32_t r, uint32_t c) const {
        return static_cast<const Derived*>(this)->get(r, c);
    }
    inline void setValue(T value) {
        std::vector<T>& vec = static_cast<Derived*>(this)->getData();
        std::fill(vec.begin(), vec.end(), value);
    }
};

template<typename T>
struct SVD {
    Matrix<T> U;
    Matrix<T> S;
    Matrix<T> V;
    SVD(Matrix<T> u, Matrix<T> s, Matrix<T> v) :
        U(u),
        S(s),
        V(v) {}
};
template<typename T>
void printMatrixMatlab(Matrix<T>& mat) {
    for (uint32_t i = 0; i < mat.rows; i++)
    {
        for (uint32_t j = 0; j < mat.cols; j++)
        {
            std::cout << mat.get(i, j);
            if (j < mat.cols - 1)
                std::cout << ", ";
        }
        std::cout << ";\n";
    }
}

template<typename A, typename DerivedA, typename DerivedB, typename DerivedR>
void matrixProduct(const MatrixInterface<A, DerivedA>& matA,
                   const MatrixInterface<A, DerivedB>& matB,
                   MatrixInterface<A, DerivedR>&       result) {

    result.setValue(A(0));
    constexpr uint32_t BS = 64;
//#ifdef _OPENMP
//    #pragma omp parallel for
//#endif
    for (int32_t jj = 0; jj < matB.getCols(); jj += BS)
        for (uint32_t kk = 0; kk < matA.getCols(); kk += BS)
            for (uint32_t ii = 0; ii < matA.getRows(); ii += BS)
                for (uint32_t j = jj; j < std::min(jj + BS, matB.getCols()); j++)
                    for (uint32_t k = kk; k < std::min(kk + BS, matA.getCols()); k++)
                    {
                        A a = matB(k, j);
                        for (uint32_t i = ii; i < std::min(ii + BS, matA.getRows()); i++)
                            result(i, j) += a * matA(i, k);
                    }
}

template<typename A>
void swapMatrices(Matrix<A>& mat1, Matrix<A>& mat2) {
    std::swap(mat1.data, mat2.data);
    std::swap(mat1.rows, mat2.rows);
    std::swap(mat1.cols, mat2.cols);
}

#endif