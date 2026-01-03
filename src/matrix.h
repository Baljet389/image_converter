#include "stdint.h"
#include <vector>
#include <iostream>

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
    uint32_t       stride;  // Physical width (NEVER swappeded)
    bool           isTransposed = false;

    Matrix(uint32_t r, uint32_t c) :
        data(r * c, T(0)),
        rows(r),
        cols(c),
        stride(c) {}
    void set(uint32_t r, uint32_t c, T value) {
        if (isTransposed)
            std::swap(r, c);
        data[r * stride + c] = value;
    }

    T get(uint32_t r, uint32_t c) const {
        if (isTransposed)
            std::swap(r, c);
        return data[r * stride + c];
    }
    void transpose() {
        isTransposed = !isTransposed;
        std::swap(rows, cols);
    }
    void setValues(std::initializer_list<T> list) {
        std::copy(list.begin(), list.end(), data.begin());
    }

    void setIdentity() {
        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                if (i == j)
                    set(i, j, T(1));
                else
                    set(i, j, T(0));
            }
        }
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

    T get(uint32_t r, uint32_t c) const { return mat.get(c, r); }

    void set(uint32_t r, uint32_t c, T value) { mat.set(c, r, value); }
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

    T get(uint32_t r, uint32_t c) const { return mat.get(r + rowOffset, c + colOffset); }

    void set(uint32_t r, uint32_t c, T value) { mat.set(r + rowOffset, c + colOffset, value); }
};
template<typename T, typename Derived>
struct MatrixInterface {
    MatrixInterface() {}

    inline uint32_t getRows() const { return static_cast<const Derived*>(this)->rows; }
    inline uint32_t getCols() const { return static_cast<const Derived*>(this)->cols; }

    inline void set(uint32_t r, uint32_t c, T value) {
        static_cast<Derived*>(this)->set(r, c, value);
    }

    inline T get(uint32_t r, uint32_t c) const {
        return static_cast<const Derived*>(this)->get(r, c);
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
    for (uint32_t i = 0; i < matA.getRows(); i++)
    {
        for (uint32_t j = 0; j < matB.getCols(); j++)
        {
            A sum = 0;
            for (uint32_t k = 0; k < matA.getCols(); k++)
            {
                sum += matA.get(i, k) * matB.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

template<typename A>
void swapMatrices(Matrix<A>& mat1, Matrix<A>& mat2) {
    std::swap(mat1.data, mat2.data);
    std::swap(mat1.rows, mat2.rows);
    std::swap(mat1.cols, mat2.cols);
    std::swap(mat1.stride, mat2.stride);
    std::swap(mat1.isTransposed, mat2.isTransposed);
}