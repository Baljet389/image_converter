#include "stdint.h"
#include <vector>


template <typename T> struct Matrix;
template <typename T> struct TransposeView;
template <typename A> struct SubMatrixView;
template <typename T> T claculateFrobeniusNorm(const Matrix<T>& mat);
 
template <typename T>
struct Matrix {
    std::vector<T> data;
    uint32_t rows, cols;

    Matrix(uint32_t r, uint32_t c) : 
    data(r * c), rows(r), cols(c) {}

    void set(uint32_t r, uint32_t c, T value) {
        data[r * cols + c] = value;
    }

    T get(uint32_t r, uint32_t c) const {
        return data[r * cols + c];
    }

    void setValues(std::initializer_list<T> list) {
        std::copy(list.begin(), list.end(), data.begin());
    }
    
    void setIdentity() {
        for(uint32_t i = 0; i < rows; i++){
            for(uint32_t j = 0; j < cols; j++){
                if(i == j) set(i, j, T(1));
                else set(i, j, T(0));
            }
        }
    }
};
template<typename T> 
struct TransposeView{
    const Matrix<T>& mat;

    TransposeView(const Matrix<T>& m) : mat(m) {}
    
    T get(uint32_t r, uint32_t c) const {
        return mat.get(c, r);
    }
    
    void set(uint32_t r, uint32_t c, T value) {
        mat.set(c, r, value);
    }
};
template<typename T> 
struct SubMatrixView{
    Matrix<T>& mat;
    uint32_t rowOffset, colOffset;
    uint32_t subRows, subCols;

    SubMatrixView(Matrix<T>& m, uint32_t rOff, uint32_t cOff, uint32_t rSize, uint32_t cSize) 
    : mat(m), rowOffset(rOff), colOffset(cOff), subRows(rSize), subCols(cSize) {}

    T get(uint32_t r, uint32_t c) const {
        return mat.get(r + rowOffset, c + colOffset);
    }

    void set(uint32_t r, uint32_t c, T value) {
        mat.set(r + rowOffset, c + colOffset, value);
    }
};

template <typename T> T claculateFrobeniusNorm(const Matrix<T>& mat){
    T sum = 0;
    for(uint32_t i = 0; i < mat.rows; i++){
        for(uint32_t j = 0; j < mat.cols; j++){
            T val = mat.get(i, j);
            sum += val * val;
        }
    }
    return std::sqrt(sum);
}