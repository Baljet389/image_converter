#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "matrix.h"
#include <string>
#include "utility.h"
#include "svd.h"
#include "qr.h"


template<typename A>
void compressImage(const std::string in, const std::string out, uint32_t rank, bool randomized);
template<typename A>
RGBImage<A> loadImage(const std::string filename);
template<typename A>
inline unsigned char clampToByte(A value);

template<typename A>
inline unsigned char clampToByte(A value) {
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return static_cast<unsigned char>(value);
}

template<typename A>
void compressImage(const std::string in, const std::string out, uint32_t rank, bool randomized) {
    RGBImage<A>             image           = loadImage<A>(in);
    std::vector<Matrix<A>>& channelMatrices = image.getChannels();

    const uint32_t             channels     = channelMatrices.size();
    const uint32_t             rows         = image.rows;
    const uint32_t             cols         = image.cols;
    const uint32_t             p            = 10;
    uint32_t                   minDimension = std::min(rows, cols);
    std::vector<unsigned char> outBuffer(rows * cols * channels);

    rank = std::min(minDimension, rank + p);

    Matrix<A> rand(cols, rank);
    Matrix<A> Y(rows, rank);

    Matrix<A> B(rank, cols);
    Matrix<A> approxB(rank, cols);
    Matrix<A> channelApprox(rows, cols);

    fillMatrixRandomValues(rand, A(0), A(1));
    for (uint32_t c = 0; c < channelMatrices.size(); c++)
    {
        Matrix<A>& mat = channelMatrices[c];

        if (randomized)
        {
            // dimensions: (rows x cols) * (cols x rank) = (rows x rank)
            matrixMultMatrix(mat, rand, Y);

            QR<A>            qr = calcQRBlocked(Y, true);
            SubMatrixView<A> QSub(qr.Q, 0, 0, rows, rank);

            // dimensions: (rank x rows) * (rows x cols) = (rank x cols)
            transposeMultMatrix(QSub, mat, B);

            SVD<A> svd = calcSVD(B);
            reconstructSVD(svd, approxB, (rank > p) ? rank - p : 0);

            // dimensions: (rows x rank) * (rank x cols) = (rows x cols)
            matrixMultMatrix(QSub, approxB, channelApprox);
        }
        else
        {
            SVD<A> svd = calcSVD(mat);
            reconstructSVD(svd, channelApprox, (rank > p) ? rank - p : 0);
        }

        for (uint32_t y = 0; y < rows; y++)
        {
            for (uint32_t x = 0; x < cols; x++)
            {
                A val = channelApprox(y, x) * static_cast<A>(255.0);

                unsigned char byteVal = clampToByte(val);
                uint32_t      outIdx  = (y * cols + x) * channels + c;
                outBuffer[outIdx]     = byteVal;
            }
        }
    }
    stbi_write_png(out.c_str(), cols, rows, channelMatrices.size(), outBuffer.data(),
                   cols * channelMatrices.size());
}
template<typename A>
RGBImage<A> loadImage(const std::string filename) {
    int            width, height, channels;
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    ASSERT(img, "Failed to load image: " + filename);

    const uint32_t rows = static_cast<uint32_t>(height);
    const uint32_t cols = static_cast<uint32_t>(width);
    const uint32_t c    = static_cast<uint32_t>(channels);

    RGBImage<A> result;
    result.rows = rows;
    result.cols = cols;

    for (uint32_t i = 0; i < c; i++)
    {
        result.channels.emplace_back(rows, cols);
    }

    for (uint32_t y = 0; y < rows; y++)
    {
        for (uint32_t x = 0; x < cols; x++)
        {
            for (uint32_t k = 0; k < c; k++)
            {
                uint32_t srcIndex        = (y * cols + x) * c + k;
                result.channels[k](y, x) = static_cast<A>(img[srcIndex]) / static_cast<A>(255.0);
            }
        }
    }
    stbi_image_free(img);
    return result;
}


#endif
