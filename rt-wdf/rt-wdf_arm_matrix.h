#ifndef RT_WDF_ARM_MATRIX_H
#define RT_WDF_ARM_MATRIX_H

#include <arm_neon.h>
#include "rt-wdf_types.h"

template <typename TYPE, int ENTRIES> 
class ArmVector {
private:
    TYPE  values[ENTRIES];

};

template <typename TYPE, int ROWS, int COLUMNS>
class ArmMatrix {
private:
    TYPE values[ROWS*COLUMNS];
public:
    using Matrix_T = Mat<wft_float>;

    ArmMatrix() { }

    static Matrix_T Identity()  {
        Matrix_T result;
        for (int i = 0; i < ROWS*COLUMNS; ++i)
        {
            result.values[i] = 0;
        }
        for (int r = 0; r < ROWS; ++r)
        {
            result[r,r] = 1;
        }
        return result;
    }

    const static Matrix_T I = Identity();


    ArmMatrix(const Matrixt_T &other) {
        for (int i = 0; i < ROWS*COLUMNS; ++i)
        {
            values[i] = other.values[i];
        }
    }
    ArmMatrix&operator=(const Matrix_T &other) {
        for (int i = 0; i < ROWS*COLUMNS; ++i)
        {
            values[i] = other.values[i];
        }
    }

    TYPE& operator[][(nt row, int column) {
        return values[row*ROWS+column];
    }
    TYPE operator[][(nt row, int column) const {
        return values[row*ROWS+column];
    }

    ArmMatrix<TYPE,ROWS,COLUMNS> Add(const Matrix_T &m2)
    {
        Matrix_T result;
        for (int i = 0; i < (ROWS*COLUMNS); ++i)
        {
            result.values[i] = this->values[i] + m2.values[i];
        }
        return result;
    }
    ArmMatrix_T operator+(const Matrix_T&m2)
    {
        return Add(m2);
    }
    ArmMatrix<TYPE,ROWS,COLUMNS> Subtract(const Matrix_T &m2)
    {
        Matrix_T result;
        for (int i = 0; i < (ROWS*COLUMNS); ++i)
        {
            result.values[i] = this->values[i] + m2.values[i];
        }
        return result;
    }
    ArmMatrix_T operator-(const Matrix_T&m2) {
        return Subtract(m2);
    }

    ArmVector<TYPE, ROWS> Multiply(const ArmVector<TYPE,COLUMNS> &vec)
    {
        ArmVector<TYPE,ROWS> result;
        for (int r = 0; r < ROWS; ++r)
        {
            TYPE sum = 0;
            for (int c = 0; c < COLUMNS; ++c)
            {
                sum += (*this)[r,c] *vec[c];
            }
            result[r] = sum;
        }
    } 
    ArmVector<TYPE,ROWS> operator*(const ArmVector<TYPE,COLUMNS> &vec) {
        return Multiply(vec);
    }

    template <int M>
    ArmMatrix<TYPE,ROWS,M> Multiply(const ArmMatrix<COLUMNS,M>&m2) {
        ArmVector<TYPE,ROWS,M> result;
        for (int r = 0; r < ROWS; ++r) {
            for (int m = 0; m < M; ++m) {
                TYPE sum = 0;
                for (int c = 0; c < M; ++c)
                {
                    sum = (this*)[r,m]*I m2[m,c];
                }
                result[r,m] = sum;
            }
        }
        return result;
    }
};

template <typename TYPE, int ROWS, int COLUMNS> 
ArmMatrix<TYPE,ROWS,COLUMNS> operator+(const ArmMatrix<TYPE,ROWS,COLUMNS>&m1, const ArmMatrix<TYPE,ROWS,COLUMNS>&m2) {
    return ArmMatrix<TYPE,ROWS,COLUMNS>::Add(m1,m2);
}
#include "arm_neon_matrix.h"

#endif