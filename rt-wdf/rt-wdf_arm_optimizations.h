/*
 *   Copyright (c) 2021 Robin E. R. Davies
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once
#ifndef RT_WDF_ARM_OPTIMIZATIONS_H
#define RT_WDF_ARM_OPTIMIZATIONS_H



#define ARM_NEON

#ifdef ARM_NEON
#include <arm_neon.h>
#include <cassert>
#include "rt-wdf_types.h"

inline float32x4_t vld1q_f32_128(const float32_t*p)
{
    return vld1q_f32((float32_t*)__builtin_assume_aligned(p,16));
}
inline float32x4_t vld1q_f32_64(const float32_t*p)
{
    return vld1q_f32((float32_t*)__builtin_assume_aligned(p,8));
}
inline void vst1q_f32_128(const float32_t*p,float32x4_t v)
{
    vst1q_f32((float32_t*)__builtin_assume_aligned(p,16),v);
}
inline void vst1q_f32_64(const float32_t*p,float32x4_t v)
{
    vst1q_f32((float32_t*)__builtin_assume_aligned(p,8),v);
}

inline void set_identity(Mat<rt_float> *m)
{
    int size = m->n_rows * m->n_cols;
    for (int i = 0; i < size; ++i)
    {
        const_cast<float_t &>(m->mem[i]) = 0;
    }
    for (int i = 0; i < m->n_rows; ++i)
    {
        (*m)(i, i) = 1;
    }
}
// Neon A32 Multiply (with reduced register set usage for A32/v7)




inline void neon_multiply(Mat<rt_float> *mResult, Mat<rt_float> *mLeft, Mat<rt_float> *mRight,
    int aRows, int aCols,int bCols
)
{
    const int bRows = aCols;

    assert(aCols == bRows);
    assert(aRows == mResult->n_rows);
    assert(bCols == mResult->n_cols);

    const float32_t *  A  ALIGNED_16  = mLeft->mem;
    const float32_t *  B  ALIGNED_16 = mRight->mem;
    float32_t *  C = (float32_t*)(mResult->mem);

    int i_row, rows;
    for (i_row = 0,rows = aRows; rows >= 4; i_row += 4,rows-=4)
    {   

        for (int i_col =0; i_col < bCols; ++i_col)
        {
            const float32_t * pB ALIGNED_16 = B + i_col*bRows;

            float32x2_t B0;
            float32x2_t B0a;

            const float32_t *   pA  ALIGNED_16  = A + i_row;

            float32x4_t A0, A1, A2, A3;

            float32x4_t C0 = vmovq_n_f32(0);

            int i_z, zeds;
            for (i_z = 0,zeds=aCols; zeds >= 4; i_z += 4,zeds-=4)
            {
                B0 = vld1_f32(pB); 
                B0a = vld1_f32(pB+2); 
                pB += 4;

                A0 = vld1q_f32_128(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                A1 = vld1q_f32_128(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A1,B0,1);

                A2 = vld1q_f32_128(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                A3 = vld1q_f32_128(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A3,B0a,1);

            }
            if ((aCols & 3) != 0) {
                if (aCols & 2) {
                    B0 = vld1_f32(pB);
                    pB += 2;

                    A0 = vld1q_f32_128(pA); pA += aRows;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);

                    A1 = vld1q_f32_128(pA); pA += aRows;
                    C0 = vmlaq_lane_f32(C0,A1,B0,1);

                    zeds -= 2;
                    i_z += 2;

                }
                if (aCols & 1)
                {
                    B0 = vmov_n_f32(*pB);
                    pB += 1;
                    A0 = vld1q_f32_128(pA); pA += aRows;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);
                    zeds -= 1;
                    i_z += 1;
                }
            }
            float32_t*pC  ALIGNED_16 = C + i_row + i_col*aRows;
            vst1q_f32_128(pC,C0);
        }
    }
    if (aRows & 0x3)
    {
        if (aRows & 0x02) {
            for (int i_col =0; i_col < bCols; ++i_col)
            {
                const float32_t *pB  ALIGNED_16 = B + i_col*bRows;

                float32x2_t B0;
                float32x2_t B0a;

                const float32_t *pA = A + i_row;

                float32x2_t A0, A1, A2, A3;

                float32x2_t C0 = vmov_n_f32(0);

                int i_z, zeds;
                for (i_z = 0,zeds=aCols; zeds >= 4; i_z += 4,zeds-=4)
                {
                    B0 = vld1_f32(pB); 
                    B0a = vld1_f32(pB+2); 
                    pB += 4;

                    A0 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A0,B0,0);

                    A1 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A1,B0,1);

                    A2 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A2,B0a,0);

                    A3 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A3,B0a,1);

                }
                if (zeds != 0) {
                    if (zeds >= 2) {
                        B0 = vld1_f32(pB);
                        pB += 2;

                        A0 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A0,B0,0);

                        A1 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A1,B0,1);

                        zeds -= 2;
                        i_z += 2;

                    }
                    if (zeds != 0)
                    {
                        B0 = vld1_f32(pB);
                        pB += 1;
                        A0 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A0,B0,0);
                        zeds -= 1;
                        i_z += 1;
                    }
                }
                float32_t*pC = C + i_row + i_col*aRows;
                vst1_f32(pC,C0);
            }

            rows -= 2;
            i_row += 2;
        }
        if (aRows & 0x1) {
            // one last row. use non-vector math.
            for (int i_col =0; i_col < bCols; ++i_col)
            {
                const float32_t *pB = B + i_col*bRows;


                const float32_t *pA = A + i_row;

                float32_t A0, A1, A2, A3;

                float32_t C0 = 0;

                int i_z;
                for (i_z = 0; i_z < aCols; ++i_z)
                {
                    C0 += (*pA)*(*pB);
                    pB += 1;
                    pA += aRows;
                }
                
                float32_t*pC = C + i_row + i_col*aRows;
                *pC = C0;
            }

            rows -= 2;
            i_row += 1;
        }

    }
}
inline void neon_matrix_x_vec(Col<rt_float> *mResult, Mat<rt_float> *mLeft, Col<rt_float> *mRight,
    int aRows, int aCols
)
{
    const int bRows = aCols;
    const int bCols = 1;

    assert(aCols == bRows);
    assert(aRows == mResult->n_rows);
    assert(bCols == mResult->n_cols);
    assert(mRight->n_cols == 1);

    const float32_t * ALIGNED_16 A = mLeft->mem;
    const float32_t * ALIGNED_16 B = mRight->mem;
    float32_t * ALIGNED_16 C = (float32_t*)(mResult->mem);

    int i_row, rows;
    int R4 = aRows & ~3;
    for (i_row = 0; i_row < R4; i_row += 4)
    {   

        {
            const float32_t * ALIGNED_16 pB = B;

            float32x2_t B0;
            float32x2_t B0a;

            const float32_t ALIGNED_16 *pA = A + i_row;

            float32x4_t A0, A1, A2, A3;

            float32x4_t C0 = vmovq_n_f32(0);

            int Z4 = aCols & ~3;
            for (int i_z = 0; i_z < Z4; i_z += 4)
            {
                B0 = vld1_f32(pB); 
                B0a = vld1_f32(pB+2); 
                pB += 4;

                A0 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                A1 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A1,B0,1);

                A2 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                A3 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A3,B0a,1);

            }
            if (aCols & 2) {
                B0 = vld1_f32(pB);
                pB += 2;

                A0 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                A1 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A1,B0,1);

            }
            if (aCols & 1)
            {
                B0 = vmov_n_f32(*pB);
                pB += 1;
                A0 = vld1q_f32(pA); pA += aRows;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);
            }
            float32_t* ALIGNED_16 pC = C + i_row;
            vst1q_f32(pC,C0);
        }
    }
    if (aRows & 0x3)
    {
        if (aRows & 0x02) {
            {
                const float32_t *pB = B;

                float32x2_t B0;
                float32x2_t B0a;

                const float32_t *pA = A + i_row;

                float32x2_t A0, A1, A2, A3;

                float32x2_t C0 = vmov_n_f32(0);

                int Z4 = aCols & ~3;
                for (int i_z = 0; i_z < Z4; i_z += 4)
                {
                    B0 = vld1_f32(pB); 
                    B0a = vld1_f32(pB+2); 
                    pB += 4;

                    A0 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A0,B0,0);

                    A1 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A1,B0,1);

                    A2 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A2,B0a,0);

                    A3 = vld1_f32(pA); pA += aRows;
                    C0 = vmla_lane_f32(C0,A3,B0a,1);

                }
                if (aCols & 3) {
                    if (aCols & 2) {
                        B0 = vld1_f32(pB);
                        pB += 2;

                        A0 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A0,B0,0);

                        A1 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A1,B0,1);

                    }
                    if (aCols & 1)
                    {
                        B0 = vld1_f32(pB);
                        pB += 1;
                        A0 = vld1_f32(pA); pA += aRows;
                        C0 = vmla_lane_f32(C0,A0,B0,0);
                    }
                }
                float32_t*pC = C + i_row;
                vst1_f32(pC,C0);
            }

            rows -= 2;
            i_row += 2;
        }
        if (aRows & 0x1) {
            // one last row. use non-vector math.
            
            {
                const float32_t *pB = B;


                const float32_t *pA = A + i_row;

                float32_t A0, A1, A2, A3;

                float32_t C0 = 0;

                int i_z;
                for (i_z = 0; i_z < aCols; ++i_z)
                {
                    C0 += (*pA)*(*pB);
                    pB += 1;
                    pA += aRows;
                }
                
                float32_t*pC = C + i_row;
                *pC = C0;
            }

            rows -= 2;
            i_row += 1;
        }

    }
}

inline void neon_multiply(Mat<rt_float> *mResult, Mat<rt_float> *mLeft, Mat<rt_float> *mRight)
{
    assert(mLeft->n_cols == mRight->n_rows);
    neon_multiply(mResult,mLeft,mRight,mLeft->n_rows,mLeft->n_cols,mRight->n_cols);
}

inline void neon_assert_vec8(Col<rt_float>*m)
{
    assert(m->n_rows == 8 && m->n_cols == 1);
}
inline void neon_add_vec8(Col<rt_float>*dest, Col<rt_float>*v1, Col<rt_float>*v2)
{
    neon_assert_vec8(dest);
    neon_assert_vec8(v1);
    neon_assert_vec8(v2);

    float32_t*pDest = dest->memptr();
    float32_t *pV1 = v1->memptr();
    float32_t *pV2 = v2->memptr();
    float32x4_t V1a = vld1q_f32_64(pV1);
    float32x4_t V1b = vld1q_f32_64(pV1+4);
    float32x4_t V2a = vld1q_f32_64(pV2);
    float32x4_t V2b = vld1q_f32_64(pV2+4);
    V1a = vaddq_f32(V1a,V2a);
    V1b = vaddq_f32(V1b,V2b);
    vst1q_f32_64(pDest,V1a);
    vst1q_f32_64(pDest+4,V1b);
}

inline void neon_assert_vec17(Col<rt_float>*m)
{
    assert(m->n_rows == 17 && m->n_cols == 1);
}

inline void neon_add_vec_17(Col<rt_float>*dest, Col<rt_float>*v1, Col<rt_float>*v2)
{
    neon_assert_vec17(dest);
    neon_assert_vec17(v1);
    neon_assert_vec17(v2);

    float32_t*pDest = dest->memptr();
    float32_t *pV1 = v1->memptr();
    float32_t *pV2 = v2->memptr();
    {
        float32x4_t V1a = vld1q_f32_64(pV1);
        float32x4_t V1b = vld1q_f32_64(pV1+4);
        float32x4_t V2a = vld1q_f32_64(pV2);
        float32x4_t V2b = vld1q_f32_64(pV2+4);
        V1a = vaddq_f32(V1a,V2a);
        V1b = vaddq_f32(V1b,V2b);
        vst1q_f32_64(pDest,V1a);
        vst1q_f32_64(pDest+4,V1b);
    }
    {
        float32x4_t V1a = vld1q_f32_64(pV1+8);
        float32x4_t V1b = vld1q_f32_64(pV1+12);
        float32x4_t V2a = vld1q_f32_64(pV2+8);
        float32x4_t V2b = vld1q_f32_64(pV2+12);
        V1a = vaddq_f32(V1a,V2a);
        V1b = vaddq_f32(V1b,V2b);
        vst1q_f32_64(pDest+8,V1a);
        vst1q_f32_64(pDest+12,V1b);
    }
    pDest[16] = pV1[16]+pV2[16];
}

inline void neon_sub_vec8(Col<rt_float>*dest, Col<rt_float>*v1, Col<rt_float>*v2)
{
    neon_assert_vec8(dest);
    neon_assert_vec8(v1);
    neon_assert_vec8(v2);

    float32_t*pDest = dest->memptr();
    float32_t *pV1 = v1->memptr();
    float32_t *pV2 = v2->memptr();
    float32x4_t V1a = vld1q_f32_64(pV1);
    float32x4_t V1b = vld1q_f32_64(pV1+4);
    float32x4_t V2a = vld1q_f32_64(pV2);
    float32x4_t V2b = vld1q_f32_64(pV2+4);
    V1a = vsubq_f32(V1a,V2a);
    V1b = vsubq_f32(V1b,V2b);
    vst1q_f32_64(pDest,V1a);
    vst1q_f32_64(pDest+4,V1b);
}

inline void neon_zero(Mat<rt_float>*m)
{
    float32x4_t ZERO = vmovq_n_f32(0);
    int n = (m->n_cols*m->n_rows)/4;
    float32_t *pMem ALIGNED_16 = m->memptr();
    for (int i = 0; i < n; ++i)
    {
        vst1q_f32_128(pMem,ZERO);
        pMem += 4;
    }
}

inline void neon_multiply_881(Col<rt_float> *mResult, Mat<rt_float> *mLeft, Col<rt_float> *mRight
)
{
    constexpr int aRows = 8;
    constexpr int aCols = 8;
    constexpr int bCols = 1;
    constexpr int bRows = 8;

    assert(aCols == bRows);
    assert(aRows == mResult->n_rows);
    assert(bCols == mResult->n_cols);
 
    const float32_t *  A ALIGNED_16= mLeft->mem;
    const float32_t * B ALIGNED_16 = mRight->mem;
    float32_t *C = (float32_t*)(mResult->mem);

    int i_row;



    const float32_t *  pB ALIGNED_8= B;

    float32x2_t B0 = vld1_f32(pB); 
    float32x2_t B0a = vld1_f32(pB+2); 
    float32x2_t B1 = vld1_f32(pB+4); 
    float32x2_t B1a = vld1_f32(pB+6); 

        {

            const float32_t * pA ALIGNED_16 = A;


            float32x4_t C0 = vmovq_n_f32(0);
            float32x4_t C1 = vmovq_n_f32(0);

            float32x4_t A0, A1, A2, A3;


            A0 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A0,B0,0);

            A1 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A1,B0,1);

            A2 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A2,B0a,0);

            A3 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A3,B0a,1);


            A0 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A0,B1,0);

            A1 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A1,B1,1);

            A2 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A2,B1a,0);

            A3 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A3,B1a,1);

            float32_t*pC ALIGNED_16 = C;
            C0 = vaddq_f32(C0,C1);
            vst1q_f32_64(pC,C0);
        }
        {

            const float32_t * pA ALIGNED_16 = A + 4;


            float32x4_t C0 = vmovq_n_f32(0);
            float32x4_t C1 = vmovq_n_f32(0);

            float32x4_t A0, A1, A2, A3;


            A0 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A0,B0,0);

            A1 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A1,B0,1);

            A2 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A2,B0a,0);

            A3 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A3,B0a,1);


            A0 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A0,B1,0);

            A1 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A1,B1,1);

            A2 = vld1q_f32_128(pA); pA += aRows;
            C0 = vmlaq_lane_f32(C0,A2,B1a,0);

            A3 = vld1q_f32_128(pA); pA += aRows;
            C1 = vmlaq_lane_f32(C1,A3,B1a,1);

            float32_t*pC ALIGNED_16 = C + 4;
            C0 = vaddq_f32(C0,C1);
            vst1q_f32_64(pC,C0);
        }
}


inline void neon_invert(Mat<rt_float> *m, Mat<rt_float> *inv)
{
    // gauss-jordan.
    // Actually transpose(transpose(A^1)) to improve access order.
    set_identity(inv);

    // reduce to upper triangular (note that we're working with  transposed data! ...  r <=> c)
    int r, c, r2, c2;
    const int nRows = 8;
    const int nCols = 8;
    for (int r = 0; r < nRows; ++r)
    {
        float *pr0 = m->colptr(r) + r;
        float *prInv0 = inv->colptr(r);

        // find the pivot row.
        float *pCol = pr0;

        int pivotRow = r;
        float maxVal = std::abs(*pCol);

        pCol += nRows;
        for (int c = r + 1; c < nRows; ++c)
        {

            float t = std::abs(*pCol);
            pCol += nRows;
            if (t > maxVal)
            {
                maxVal = t;
                pivotRow = c;
            }
        }
        if (maxVal == 0)
            throw std::invalid_argument("Matrix is non-invertable");

        if (pivotRow != r)
        {
            // swap the pivot row and normalize it.

            float *pr = pr0;
            float *prPiv = m->colptr(pivotRow) + r;
            float pivMul = 1 / *prPiv;

            *prPiv++ = *pr;
            *pr++ = 1;

            for (int c = r + 1; c < nRows; ++c)
            {
                float t = *pr;
                *pr = pivMul * *prPiv;
                *prPiv = t;
                ++pr;
                ++prPiv;
            }
            float *prInv = prInv0;
            float *prPivInv = inv->colptr(pivotRow);
            for (int c = 0; c < nRows; ++c)
            {
                float t = *prInv;
                *prInv = (*prPivInv) * pivMul;
                *prPivInv = t;
                ++prInv;
                ++prPivInv;
            }
        }
        else
        {
            // normalize the pivot row (which is already in the right place)
            float *pr = pr0;
            float pivMul = 1 / *pr;
            *pr++ = 1;

            for (int c = r + 1; c < nRows; ++c)
            {
                *pr++ *= pivMul;
            }
            float *prInv = prInv0;
            for (int c = 0; c < nRows; ++c)
            {
                *prInv++ *= pivMul;
            }
        }
        for (int r2 = r + 1; r2 < nRows; ++r2)
        {
            float *pr = pr0;
            float *pr2 = m->colptr(r2) + r;
            if (*pr2 != 0)
            {
                float factor = -*pr2;
                *pr2 = 0;
                ++pr2;
                ++pr;
                for (int c = r + 1; c < nRows; ++c)
                {
                    *pr2++ += factor * (*pr++);
                }
                float *prInv = prInv0;
                float *prInv2 = inv->colptr(r2);
                for (int c = 0; c < nRows; ++c)
                {
                    *prInv2++ += *prInv++ * factor;
                }
            }
        }
    }

    // backsolve the upper triangle.

    for (int r = nRows - 1; r >= 1; --r)
    {
        float *prInvSrc0 = inv->colptr(r);
        float *prInvDest = inv->colptr(0);
        float *prDest = m->colptr(0) + r;
        for (int r2 = 0; r2 < r; ++r2)
        {
            float *prInvSrc = prInvSrc0;

            float val = *prDest;
            prDest += nRows;

            // *prDest = 0;  (we won't use it again)
            for (int c = 0; c < nRows; ++c)
            {
                *prInvDest++ -= val * *prInvSrc++;
            }
        }
    }
}


inline void neon_identity_88(Mat<rt_float>*m)
{
    float32_t*  ALIGNED_16 p = m->memptr();
    float32x4_t ZERO = vmovq_n_f32(0);
    float32x4_t ONE = vsetq_lane_f32(1,ZERO,0);

    vst1q_f32_128(p,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    vst1q_f32_128(p+4,ZERO); 
    p += 8;

    vst1q_f32_128(p,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    vst1q_f32_128(p+4,ZERO); 
    p += 8;

    vst1q_f32_128(p,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    vst1q_f32_128(p+4,ZERO); 
    p += 8;

    vst1q_f32_128(p,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    vst1q_f32_128(p+4,ZERO); 
    p += 8;

    vst1q_f32_128(p,ZERO); 
    vst1q_f32_128(p+4,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    p += 8;

    vst1q_f32_128(p,ZERO); 
    vst1q_f32_128(p+4,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    p += 8;

    vst1q_f32_128(p,ZERO); 
    vst1q_f32_128(p+4,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    p += 8;
    vst1q_f32_128(p,ZERO); 
    vst1q_f32_128(p+4,ONE); 
    ONE = vextq_f32(ONE,ONE,3);
    p += 8;
}

inline void neon_invert_88(Mat<rt_float> *m, Mat<rt_float> *inv)
{
    // gauss-jordan.

    // to improve memory access order, we pretend that the matrices are in row-major format
    // even though they are in column-major format. Because transpose((transponse(A)^1)) = A^1
    // this just works. From hereon, terminology and coments assume row-major format.
    neon_identity_88(inv);

    // reduce to upper triangular format 
    int r, c, r2, c2;
    const int nRows = 8;  // e.g. rows are actually columns from the column-major perspective. &c.
    const int nCols = 8;
    for (int r = 0; r < nRows; ++r)
    {
        float * ALIGNED_16 pPivot = m->colptr(r);
        float * ALIGNED_16 pInvPivot = inv->colptr(r);

        // find the pivot row.
        float *pColValue = pPivot +r;


        float pivotValue = *pColValue;
        float bestValue = std::abs(pivotValue);
        int pivotRow = r;

        pColValue += nRows;
        for (int r2 = r + 1; r2 < nRows; ++r2)
        {
            float t = *pColValue;
            float absT = std::abs(t);
            pColValue += nRows;
            if (absT > bestValue)
            {
                bestValue = absT;
                pivotValue = t;
                pivotRow = r2;
            }
        }
        if (bestValue == 0)
            throw std::invalid_argument("Matrix is non-invertable");

        float pivMul = 1 / pivotValue;

        float32x4_t INV = vmovq_n_f32(pivMul);


        if (true) //if (r < 4)   // ~1% speed-up for  12.5% reduction in loads/stores.
        {

            float32x4_t T0;
            float32x4_t T1;
            float32x4_t TAUX0;
            float32x4_t TAUX1;
            if (pivotRow != r)
            {
                // swap the pivot row and normalize it.
                float * ALIGNED_16 pSrc = m->colptr(pivotRow);

                T0 = vld1q_f32_128(pSrc);
                T1 = vld1q_f32_128(pSrc+4);
                float32x4_t S0 = vld1q_f32_128(pPivot);
                float32x4_t S1 = vld1q_f32_128(pPivot+4);

                vst1q_f32_128(pSrc,S0);
                vst1q_f32_128(pSrc+4,S1);

                T0 = vmulq_f32(T0,INV);
                //T0 = vsetq_lane_f32(1,T0,r);
                T1 = vmulq_f32(T1,INV);
                vst1q_f32_128(pPivot,T0);
                vst1q_f32_128(pPivot+4,T1);
                pPivot[r] = 1; // instead of vsetq. Is this a serious stall?


                float *pInvSrc = inv->colptr(pivotRow);
                float *pInvPivot = inv->colptr(r);
                // now on the aux matrix.
                TAUX0 = vld1q_f32_128(pInvSrc);
                TAUX1 = vld1q_f32_128(pInvSrc+4);
                S0 = vld1q_f32_128(pInvPivot);
                S1 = vld1q_f32_128(pInvPivot+4);

                vst1q_f32_128(pInvSrc,S0);
                vst1q_f32_128(pInvSrc+4,S1);

                TAUX0 = vmulq_f32(TAUX0,INV);
                TAUX1 = vmulq_f32(TAUX1,INV);
                vst1q_f32_128(pInvPivot,TAUX0);
                vst1q_f32_128(pInvPivot+4,TAUX1);
            }
            else
            {
                T0 = vld1q_f32_128(pPivot);
                T1 = vld1q_f32_128(pPivot+4);
                T0 = vmulq_f32(T0,INV);
                T1 = vmulq_f32(T1,INV);
                //T0 = vsetq_lane_f32(1,T0,r);

                vst1q_f32_128(pPivot,T0);
                vst1q_f32_128(pPivot+4,T1);
                pPivot[r] = 1;

                TAUX0 = vld1q_f32_128(pInvPivot);
                TAUX1 = vld1q_f32_128(pInvPivot+4);
                TAUX0 = vmulq_f32(TAUX0,INV);
                TAUX1 = vmulq_f32(TAUX1,INV);
                vst1q_f32_128(pInvPivot,TAUX0);
                vst1q_f32_128(pInvPivot+4,TAUX1);
            }
            // T0, T1 contain the pivot row.
            // TAUX0, TAUX1 contain the pivot row in the auxilliary matrix.

            ALIGNED_16 float32_t * pRow = pPivot+8;
            float32_t *  pInvRow ALIGNED_16 = inv->colptr(r+1);

            for (int r2 = r + 1; r2 < nRows; ++r2)
            {
                float32x4_t NEG = vmovq_n_f32(pRow[r]);
                NEG = vnegq_f32(NEG);
                float32x4_t S0 = vld1q_f32_128(pRow);
                float32x4_t S1 = vld1q_f32_128(pRow+4);
                S0 = vmlaq_f32(S0,T0,NEG); 
                //S0 = vsetq_lane_f32(0,S0,r); // clear rounding errors. should cancel perfectly.

                S1 = vmlaq_f32(S1,T1,NEG); 
                vst1q_f32_128(pRow,S0);
                vst1q_f32_128(pRow+4,S1);
                pRow[r] = 0;

                S0 = vld1q_f32_128((ALIGNED_16 float32_t *)pInvRow);
                S1 = vld1q_f32_128((float32_t*)__builtin_assume_aligned(pInvRow+4,16));
                S0 = vmlaq_f32(S0,TAUX0,NEG);
                S1 = vmlaq_f32(S1,TAUX1,NEG);
                vst1q_f32_128(pInvRow,S0);
                vst1q_f32_128(pInvRow+4,S1);

                pRow += 8;
                pInvRow += 8;
            }

        } else { 
            
            // r >= 4. Only need to touch lower right quadrant.
            float32x4_t T1;
            float32x4_t TAUX0;
            float32x4_t TAUX1;
            if (pivotRow != r)
            {
                // swap the pivot row and normalize it.
                float * ALIGNED_16 pSrc = m->colptr(pivotRow);
                float32x4_t INV = vmovq_n_f32(pivMul);

                T1 = vld1q_f32_128(pSrc+4);
                float32x4_t S1 = vld1q_f32_128(pPivot+4);

                vst1q_f32_128(pSrc+4,S1);

                T1 = vmulq_f32(T1,INV);
                //T1 = vsetq_lane_f32(1,T1,r-4);
                vst1q_f32_128(pPivot+4,T1);
                pPivot[r] = 1;

                float * ALIGNED_16 pInvSrc = inv->colptr(pivotRow);
                // now on the aux matrix.
                TAUX0 = vld1q_f32_128(pInvSrc);
                TAUX1 = vld1q_f32_128(pInvSrc+4);
                float32x4_t S0 = vld1q_f32_128(pInvPivot);
                S1 = vld1q_f32_128(pInvPivot+4);

                vst1q_f32_128(pInvSrc,S0);
                vst1q_f32_128(pInvSrc+4,S1);

                TAUX0 = vmulq_f32(TAUX0,INV);
                TAUX1 = vmulq_f32(TAUX1,INV);
                vst1q_f32_128(pInvPivot,TAUX0);
                vst1q_f32_128(pInvPivot+4,TAUX1);
            }
            else
            {
                T1 = vld1q_f32_128(pPivot+4);
                T1 = vmulq_f32(T1,INV);
                // T1 = vsetq_lane_f32(1,T1,r-4);

                vst1q_f32_128(pPivot+4,T1);
                pPivot[r] = 1;

                TAUX0 = vld1q_f32_128(pInvPivot);
                TAUX1 = vld1q_f32_128(pInvPivot+4); 
                TAUX0 = vmulq_f32(TAUX0,INV);
                TAUX1 = vmulq_f32(TAUX1,INV);
                vst1q_f32_128(pInvPivot,TAUX0);
                vst1q_f32_128(pInvPivot+4,TAUX1);
            }
            // T1 contain the pivot row.
            // TAUX0, TAUX1 contain the pivot row in the auxilliary matrix.

            float32_t *pRow = pPivot+8;
            float32_t *pInvRow = inv->colptr(r+1);

            for (int r2 = r + 1; r2 < nRows; ++r2)     // 7x8!! Could we cancel 2 entries?
            {
                float32x4_t NEG = vmovq_n_f32(pRow[r]);
                NEG = vnegq_f32(NEG);
                float32x4_t S1 = vld1q_f32_128(pRow+4);

                S1 = vmlaq_f32(S1,T1,NEG); 
                //S1 = vsetq_lane_f32(0,S1,r-4); // clear rounding errors. should cancel perfectly.

                vst1q_f32_128(pRow+4,S1);
                pRow[r] = 0;

                float32x4_t S0 = vld1q_f32_128(pInvRow);
                S1 = vld1q_f32_128(pInvRow+4);
                S0 = vmlaq_f32(S0,TAUX0,NEG);
                S1 = vmlaq_f32(S1,TAUX1,NEG);
                vst1q_f32_128(pInvRow,S0);
                vst1q_f32_128(pInvRow+4,S1);

                pRow += 8;
                pInvRow += 8;
            }
        }
    }

    // backsolve the upper triangle.

    // cancel 2 rows.
    for (int r = nRows - 1; r >= 1; r -= 2)
    {
        float * ALIGNED_16 prInvSrc0 = inv->colptr(r);
        float * ALIGNED_16 prInvDest = prInvSrc0-nRows;
        float *prDest = m->colptr(r-1) + r;

        float32x4_t INVROW0_0 = vld1q_f32_128(prInvSrc0);
        float32x4_t INVROW0_4 = vld1q_f32_128(prInvSrc0+4);
        INVROW0_0 = vnegq_f32(INVROW0_0);
        INVROW0_4 = vnegq_f32(INVROW0_4);

        float32x4_t VAL = vmovq_n_f32(*prDest);

        float32x4_t INVROW1_0 = vld1q_f32_128(prInvDest);
        float32x4_t INVROW1_4 = vld1q_f32_128(prInvDest+(4));
        INVROW1_0 = vmlaq_f32(INVROW1_0,INVROW0_0,VAL);
        INVROW1_4 = vmlaq_f32(INVROW1_4,INVROW0_4,VAL);
        vst1q_f32_128(prInvDest,INVROW1_0);
        vst1q_f32_128(prInvDest+4,INVROW1_4);
        INVROW1_0 = vnegq_f32(INVROW1_0);
        INVROW1_4 = vnegq_f32(INVROW1_4);

        prDest -= nRows;
        prInvDest -= nRows;


        // cancel 2 rows.
        for (int r2 = 1; r2 < r; r2 += 1)  //  7x4!!!  
        {
            float32x4_t VAL0 = vmovq_n_f32(*prDest);
            float32x4_t VAL1 = vmovq_n_f32(prDest[-1]);

            float32x4_t INVDST_0 = vld1q_f32_128(prInvDest);
            float32x4_t INVDST_4 = vld1q_f32_128(prInvDest+4);
            INVDST_0 = vmlaq_f32(INVDST_0,INVROW0_0,VAL0);
            INVDST_4 = vmlaq_f32(INVDST_4,INVROW0_4,VAL0);
            INVDST_0 = vmlaq_f32(INVDST_0,INVROW1_0,VAL1);
            INVDST_4 = vmlaq_f32(INVDST_4,INVROW1_4,VAL1);
            vst1q_f32_128(prInvDest,INVDST_0);
            vst1q_f32_128(prInvDest+4,INVDST_4);

            prDest -= nRows;
            prInvDest -= nRows;
        }
        
    }
}

inline void neon_matrix_x_vec(Col<rt_float> *mResult, Mat<rt_float> *mLeft, Col<rt_float> *mRight)
{
    neon_matrix_x_vec(mResult,mLeft,mRight,mLeft->n_rows,mLeft->n_cols);
}

#endif

#endif