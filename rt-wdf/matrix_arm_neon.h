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

// ==== Specialization of Vector<N,float_t> and Matrix<R,C,float_t> to Arm Neon.
#ifndef MATRIX_ARM_NEON_H_
#define MATRIX_ARM_NEON_H_

#include <arm_neon.h>

// ==== Vector<N,float32_t>  ====== 

template <int N> class Vector<N,float32_t> {
public:
    using ELEMENT_TYPE=float32_t;
private:
    ELEMENT_TYPE data[impl::padSize<ELEMENT_TYPE>(N)+MATRIX_ALIGNMENT/sizeof(ELEMENT_TYPE)];
    ELEMENT_TYPE *mem M_ALIGNED;
public: 
    using VectorT=Vector<N,ELEMENT_TYPE>;
    static constexpr int Size = N;

    Vector() {
        mem = impl::alignPointer(data);
    }

    Vector(const std::initializer_list<ELEMENT_TYPE> &values)
    :Vector()
    {
        assert(values.size() == N);

        ELEMENT_TYPE *p = mem;
        for (auto i = values.begin(); i != values.end(); ++i)
        {
            *p++ = (*i);
        }
    }
    template <typename DELAYED_OP,
        typename =decltype(DELAYED_OP::IsDelayedOp(VectorT()))
         >
    Vector(const DELAYED_OP&op)
    :Vector()
    {
        op.Apply(*this);
    }

    ELEMENT_TYPE*memory()  M_ALIGNED { return mem; }
    const ELEMENT_TYPE*memory() const  M_ALIGNED { return mem; }

    ELEMENT_TYPE &operator[](int index) { return mem[index];}
    ELEMENT_TYPE operator[](int index) const { return mem[index];}
    bool operator==(const VectorT &other);
    bool operator!=(const VectorT &other);

    void AddAssign(const VectorT&v1, const VectorT&v2);
    void SubtractAssign(const VectorT&v1, const VectorT&v2);
    void NegateAssign(const VectorT&v1);


    template <int COLUMNS>
    void MultiplyAssign(const Matrix<N,COLUMNS,ELEMENT_TYPE> &matrix,const Vector<COLUMNS,ELEMENT_TYPE> &vector);


    void Set(ELEMENT_TYPE value);

    constexpr VectorT&operator=(Vector&v);

    template <typename DELAYED_OP> 
    constexpr decltype(DELAYED_OP::IsDelayedOp(VectorT()), VectorT()) & // Must be DelayedOp<VectorT>
    operator=(const DELAYED_OP&op)
    {
        op.Apply(*this);
        return *this;
    }

};

template <int N>
constexpr Vector<N,float32_t>&Vector<N,float32_t>::operator=(Vector<N,float32_t>&v)
{
    constexpr int NC = N & ~3;
    float*pSrc M_ALIGNED = v.mem;
    float *pDst M_ALIGNED = mem;
    if (NC == 4) {
        float32x4_t T0 = vld1q_f32(pSrc);
        vst1q_f32(pDst,T0);
        pSrc += 4;
        pDst += 4;
    } else if (NC == 8)
    {
        float32x4_t T0 = vld1q_f32(pSrc);
        float32x4_t T1 = vld1q_f32(pSrc+4);
        vst1q_f32(pDst,T0);
        vst1q_f32(pDst+4,T1);
        pSrc += 8;
        pDst += 8;
    } else if (NC == 12)
    {
        float32x4_t T0 = vld1q_f32(pSrc);
        float32x4_t T1 = vld1q_f32(pSrc+4);
        float32x4_t T2 = vld1q_f32(pSrc+8);
        vst1q_f32(pDst,T0);
        vst1q_f32(pDst+4,T1);
        vst1q_f32(pDst+8,T2);
        pSrc += 12;
        pDst += 12;
    } else if (NC >= 16)
    {
        constexpr int NC8 = N & ~0x07;
        for (int i = 0; i < NC8; i += 8)
        {
            float32x4_t T0 = vld1q_f32(pSrc);
            float32x4_t T1 = vld1q_f32(pSrc+4);
            vst1q_f32(pDst,T0);
            vst1q_f32(pDst+4,T1);
            pSrc += 8;
            pDst += 8;
        }
        if (N & 0x04)
        {
            float32x4_t T0 = vld1q_f32(pSrc);
            vst1q_f32(pDst,T0);
            pSrc += 4;
            pDst += 4;
        }
    }
    if (N & 0x02)
    {
        float32x2_t T0 = vld1_f32(pSrc);
        vst1_f32(pDst,T0);
        pSrc += 2;
        pDst += 2;
    }
    if (N & 0x01)
    {
        float32_t t = *pSrc;
        *pDst = t;
    }
    return *this;
}

template <int N>
void Vector<N,float32_t>::AddAssign(const VectorT&v1, const VectorT&v2)
{
    constexpr int NC = N & ~3;
    float*pSrc1 M_ALIGNED = v1.mem;
    float*pSrc2 M_ALIGNED = v2.mem;
    float *pDst M_ALIGNED = mem;
    if (NC < 16) {
        if (NC >= 4) {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
        if (NC >= 8)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        } 
        if (NC == 12)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
    } else // if (NC >= 16)
    {
        constexpr int NC8 = N & ~0x07;
        for (int i = 0; i < NC8; i += 8)
        {
            {
                float32x4_t T0s1 = vld1q_f32(pSrc1);
                float32x4_t T0s2 = vld1q_f32(pSrc2);
                vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
                pSrc1 += 4;
                pSrc2 += 4;
                pDst += 4;
            }

            {
                float32x4_t T0s1 = vld1q_f32(pSrc1);
                float32x4_t T0s2 = vld1q_f32(pSrc2);
                vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
                pSrc1 += 4;
                pSrc2 += 4;
                pDst += 4;
            }
        }
        if (N & 0x04)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vaddq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
    }
    if (N & 0x02)
    {
        float32x2_t T0s1 = vld1_f32(pSrc1);
        float32x2_t T0s2 = vld1_f32(pSrc2);
        vst1_f32(pDst,vadd_f32(T0s1,T0s2));
        pSrc1 += 2;
        pSrc2 += 2;
        pDst += 2;
    }
    if (N & 0x01)
    {
        float32_t t = *pSrc1 + *pSrc2;
        *pDst = t;
    }
}

template <int N>
void Vector<N,float32_t>::SubtractAssign(const VectorT&v1, const VectorT&v2)
{
    constexpr int NC = N & ~3;
    float*pSrc1 M_ALIGNED = v1.mem;
    float*pSrc2 M_ALIGNED = v2.mem;
    float *pDst M_ALIGNED = mem;
    if (NC < 16) {
        if (NC >= 4) {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
        if (NC >= 8)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        } 
        if (NC == 12)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
    } else // if (NC >= 16)
    {
        constexpr int NC8 = N & ~0x07;
        for (int i = 0; i < NC8; i += 8)
        {
            {
                float32x4_t T0s1 = vld1q_f32(pSrc1);
                float32x4_t T0s2 = vld1q_f32(pSrc2);
                vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
                pSrc1 += 4;
                pSrc2 += 4;
                pDst += 4;
            }

            {
                float32x4_t T0s1 = vld1q_f32(pSrc1);
                float32x4_t T0s2 = vld1q_f32(pSrc2);
                vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
                pSrc1 += 4;
                pSrc2 += 4;
                pDst += 4;
            }
        }
        if (N & 0x04)
        {
            float32x4_t T0s1 = vld1q_f32(pSrc1);
            float32x4_t T0s2 = vld1q_f32(pSrc2);
            vst1q_f32(pDst,vsubq_f32(T0s1,T0s2));
            pSrc1 += 4;
            pSrc2 += 4;
            pDst += 4;
        }
    }
    if (N & 0x02)
    {
        float32x2_t T0s1 = vld1_f32(pSrc1);
        float32x2_t T0s2 = vld1_f32(pSrc2);
        vst1_f32(pDst,vsub_f32(T0s1,T0s2));
        pSrc1 += 2;
        pSrc2 += 2;
        pDst += 2;
    }
    if (N & 0x01)
    {
        float32_t t = *pSrc1 - *pSrc2;
        *pDst = t;
    }
}
template <int N>
void Vector<N,float32_t>::NegateAssign(const VectorT&v1)
{
    static constexpr int N4 = N & ~0x03;
    float32_t *mem = v1.mem;

    for (int i = 0; i < N; ++i)
    {
        (*this)[i] = -v1[i];
    }
}
template<int N>
template<int COLUMNS>
void Vector<N,float32_t>::MultiplyAssign(const Matrix<N,COLUMNS,float32_t> &m,const Vector<COLUMNS,float32_t> &v)
{
    constexpr int aRows = N;
    constexpr int aCols = COLUMNS;
    constexpr int bRows = aCols;
    constexpr int bCols = 1;
    constexpr int aStride = m.STRIDE;


    const float32_t *  A M_ALIGNED = v.mem;
    const float32_t *  B M_ALIGNED = m.mem;
    float32_t *  C M_ALIGNED = (this->mem);

    constexpr int R4 = N & ~0x03;
    constexpr int Z = COLUMNS;
    constexpr int Z4 = Z & ~0x03;

    int i_row;
    for (i_row = 0; i_row < R4; i_row += 4)
    {   
        const float32_t * pB M_ALIGNED = B;

        float32x2_t B0;
        float32x2_t B0a;

        const float32_t *pA M_ALIGNED = A + i_row;

        float32x4_t A0, A1, A2, A3;

        float32x4_t C0 = vmovq_n_f32(0);
        float32x4_t C1 = vmovq_n_f32(0);

        if (Z4 < 16) {
            if (Z4 >= 4) {
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);

                float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A3,B0a,1);
            }
            if (Z4 >= 8) {
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);

                float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A3,B0a,1);
            }
            if (Z4 >= 12) {
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);

                float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A3,B0a,1);
            }

        } else  // Z4 >= 16.
        {
            for (int z = 0; z < Z4; z += 4)
            {
            
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);

                float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A3,B0a,1);
            }
            if (Z & 0x04) {
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);

                float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A3,B0a,1);
            }
        }
        if (Z & 2)
        {
            B0 = vld1_f32(pB);
            pB += 2;

            A0 = vld1q_f32(pA); pA += aStride;
            C0 = vmlaq_lane_f32(C0,A0,B0,0);

            A1 = vld1q_f32(pA); pA += aStride;
            C1 = vmlaq_lane_f32(C1,A1,B0,1);
        }
        if (Z & 1)
        {
            B0 = vmov_n_f32(*pB);
            pB += 1;
            A0 = vld1q_f32(pA); pA += aStride;
            C0 = vmlaq_lane_f32(C0,A0,B0,0);
        }
        C0 = vaddq_f32(C0,C1);
        float32_t* pC M_ALIGNED = C + i_row;
        vst1q_f32(pC,C0);
    }
    if (aRows & 0x02) {
        const float32_t *pB = B;

        float32x2_t B0;

        const float32_t *pA M_ALIGNED = A + i_row;

        float32x2_t C0 = vmov_n_f32(0);
        float32x2_t C1 = vmov_n_f32(0);

        int i_z;
        for (i_z = 0; i_z < Z4; i_z += 4)
        {
            float32x2_t B0 = vld1_f32(pB); 
            float32x2_t B0a = vld1_f32(pB+2); 
            pB += 4;

            float32x2_t A0 = vld1_f32(pA); pA += aStride;
            C0 = vmla_lane_f32(C0,A0,B0,0);

            float32x2_t A1 = vld1_f32(pA); pA += aStride;
            C1 = vmla_lane_f32(C1,A1,B0,1);

            float32x2_t A2 = vld1_f32(pA); pA += aStride;
            C0 = vmla_lane_f32(C0,A2,B0a,0);

            float32x2_t A3 = vld1_f32(pA); pA += aStride;
            C1 = vmla_lane_f32(C1,A3,B0a,1);

        }
        if (Z & 2) {
            float32x2_t B0 = vld1_f32(pB);
            pB += 2;

            float32x2_t A0 = vld1_f32(pA); pA += aStride;
            C0 = vmla_lane_f32(C0,A0,B0,0);

            float32x2_t A1 = vld1_f32(pA); pA += aStride;
            C1 = vmla_lane_f32(C1,A1,B0,1);
        }
        if (Z & 1)
        {
            float32x2_t B0 = vmov_n_f32(*pB);
            pB += 1;
            float32x2_t A0 = vld1_f32(pA); pA += aStride;
            C0 = vmla_lane_f32(C0,A0,B0,0);
        }
        C0 = vadd_f32(C0,C1);
        float32_t*pC = C + i_row;
        vst1_f32(pC,C0);
    }
    if (aRows & 0x1) {
        // one last row. use non-vector math.
        
        {
            const float32_t *pB = B;


            const float32_t *pA = A + i_row;

            float32_t C0 = 0;

            int i_z;
            for (i_z = 0; i_z < aCols; ++i_z)
            {
                C0 += (*pA)*(*pB);
                pB += 1;
                pA += aStride;
            }
            
            float32_t*pC = C + i_row;
            *pC = C0;
        }
    }
}
template <int N>
void Vector<N,float32_t>::Set(ELEMENT_TYPE value)
{
    for (int i = 0; i < this->Size; ++i) mem[i] = value;
}
template <int N>
bool Vector<N,float32_t>::operator==(const VectorT &other)
{
    for (int i = 0; i < Size; ++i)
    {
        if (mem[i] != other[i]) return false;
    }
    return true;
}
template <int N>
bool Vector<N,float32_t>::operator!=(const VectorT &other)
{
    return !(*this == other);
}


//=========== Matrix specialization =================/

template <int ROWS, int COLUMNS> 
class Matrix<ROWS,COLUMNS,float32_t> {
public:
    using ELEMENT_TYPE=float32_t;

    static constexpr int STRIDE = impl::padSize<ELEMENT_TYPE>(ROWS);
private:
    ELEMENT_TYPE data[STRIDE*COLUMNS+MATRIX_ALIGNMENT/sizeof(ELEMENT_TYPE)];
    ELEMENT_TYPE *mem M_ALIGNED;
public:
    static constexpr int Rows = ROWS;
    static constexpr int Columns = COLUMNS;
    using ElementTypeT = ELEMENT_TYPE;

    using MatrixT = Matrix<ROWS,COLUMNS,ELEMENT_TYPE>;

    Matrix() {
        mem = impl::alignPointer<ELEMENT_TYPE>(data);
    }

    Matrix(const std::initializer_list<std::initializer_list<ELEMENT_TYPE> > &list)
    :Matrix()
    {
        assert(list.size() == ROWS);

        int ir = 0;
        for (auto r = list.begin(); r != list.end(); ++r,++ir)
        {
            const std::initializer_list<ELEMENT_TYPE>&row = (*r);
            assert(row.size() == COLUMNS);

            int ic = 0;
            for (auto c = row.begin(); c != row.end(); ++c,++ic)
            {
                ELEMENT_TYPE value = (*c);
                (*this)(ir,ic) = value;
            }
        }
    }
    template <typename DELAYED_OP,
        typename =decltype(DELAYED_OP::IsDelayedOp(MatrixT()))
        >
    Matrix(const DELAYED_OP&op)
    :Matrix()
    {
        op.Apply(*this);
    }


    ELEMENT_TYPE& operator()(int row, int column) { 
        assert(row >= 0 && row < Rows);
        assert(column >= 0 && column < Columns);
        int index = column*STRIDE+row;
        assert(index < sizeof(data)/sizeof(data[0]));
        return mem[index]; 
    }
    ELEMENT_TYPE operator()(int row, int column) const { 
        assert(row >= 0 && row < Rows);
        assert(column >= 0 && column < Columns);
        int index = column*STRIDE+row;
        assert(index < sizeof(data)/sizeof(data[0]));
        return mem[index]; 
    }
    ELEMENT_TYPE *memory()  {
        return mem;
    }
    const ELEMENT_TYPE *memory() const  {
        return mem;
    }
    ELEMENT_TYPE* columnAddress(int column)   { 
        assert(column >= 0 && column < COLUMNS);
        return &mem[column*STRIDE];
    }
    const ELEMENT_TYPE*columnAddress(int column) const   { 
        assert(column >= 0 && column < COLUMNS);
        return &mem[column*STRIDE];
    }

    bool operator==(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other);
    bool operator!=(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other);


    void AddAssign(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m1, const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m2);

    void SubtractAssign(const MatrixT&m1, const MatrixT &m2);

    template <int Z>
    void MultiplyAssign(const Matrix<Rows,Z,ELEMENT_TYPE> &m1, const Matrix<Z,Columns,ELEMENT_TYPE> &m2);

    void InvertAssign(const Matrix<ROWS,COLUMNS,float_t>&m1);

    Matrix<ROWS,COLUMNS,ELEMENT_TYPE>&operator=(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE>&other);

    template<typename DELAYED_MATRIX_OP
    >
    Matrix<ROWS,COLUMNS,ELEMENT_TYPE>&operator=(const DELAYED_MATRIX_OP &op)
    {
        op.Apply(*this);
        return *this;
    }

    std::string toString();

};


template <int ROWS, int COLUMNS> 
Matrix<ROWS,COLUMNS,float32_t>&Matrix<ROWS,COLUMNS,float32_t>::operator=(const Matrix<ROWS,COLUMNS,float32_t>&other)
{
    int n = STRIDE*Rows;
    float32_t *src M_ALIGNED = other.mem;
    float32_t *dst M_ALIGNED = mem;
    for (int i = 0; i < n; i += 4)
    {
        float32x4_t T0 = vld1q_f32(src+i);
        vst1q_f32(dst+i,T0);
    }
    return *this;
}



template <int ROWS, int COLUMNS>
void Matrix<ROWS,COLUMNS,float32_t>::AddAssign(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m1, const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m2)
{
    for (int c = 0; c < COLUMNS; ++c) 
    {
        constexpr int R4 = ROWS & ~0x03;
        float32_t*pSrc1 M_ALIGNED = m1.columnAddress(c);
        float32_t*pSrc2 M_ALIGNED = m2.columnAddress(c);
        float32_t *pDest M_ALIGNED = this->columnAddress(c);
        if (R4 < 16)
        {
            if (R4 >= 4)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vaddq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
            if (R4 >= 8)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vaddq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
            if (R4 >= 12)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vaddq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }

        } else {
            for (int r = 0; r < R4; r += 4) 
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vaddq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
        }
        if (ROWS & 2) {
            float32x2_t T0 = vld1_f32(pSrc1);
            float32x2_t T1 = vld1_f32(pSrc2);
            vst1_f32(pDest,vadd_f32(T0,T1));
            pSrc1 += 2; pSrc2 += 2; pDest += 2;
        }
        if (ROWS & 1) 
        {
            *pDest =  *pSrc1 + *pSrc2;
        }
    }
}

template <int ROWS, int COLUMNS>
void Matrix<ROWS,COLUMNS,float32_t>::SubtractAssign(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m1, const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m2)
{
    for (int c = 0; c < COLUMNS; ++c) 
    {
        constexpr int R4 = ROWS & ~0x03;
        float32_t*pSrc1 M_ALIGNED = m1.columnAddress(c);
        float32_t*pSrc2 M_ALIGNED = m2.columnAddress(c);
        float32_t *pDest M_ALIGNED = this->columnAddress(c);
        if (R4 < 16)
        {
            if (R4 >= 4)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vsubq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
            if (R4 >= 8)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vsubq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
            if (R4 >= 12)
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vsubq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }

        } else {
            for (int r = 0; r < R4; r += 4) 
            {
                float32x4_t T0 = vld1q_f32(pSrc1);
                float32x4_t T1 = vld1q_f32(pSrc2);
                vst1q_f32(pDest,vsubq_f32(T0,T1));
                pSrc1 += 4; pSrc2 += 4; pDest += 4;
            }
        }
        if (ROWS & 2) {
            float32x2_t T0 = vld1_f32(pSrc1);
            float32x2_t T1 = vld1_f32(pSrc2);
            vst1_f32(pDest,vsub_f32(T0,T1));
            pSrc1 += 2; pSrc2 += 2; pDest += 2;
        }
        if (ROWS & 1) 
        {
            *pDest = *pSrc1- *pSrc2;
        }
    }
}

template <int ROWS, int COLUMNS>
template <int Z>
void Matrix<ROWS,COLUMNS,float32_t>::MultiplyAssign(const Matrix<Rows,Z,ELEMENT_TYPE> &m1, const Matrix<Z,Columns,ELEMENT_TYPE> &m2)
{
    assert((void*)this != (void*)&m1 && (void*)this != (void*)&m2); // cannot alias matrices when multiplying.

    constexpr int aRows = ROWS;
    constexpr int aCols = Z;
    constexpr int bRows = Z;
    constexpr int aStride = m1.STRIDE;
    constexpr int bCols = COLUMNS;
    constexpr int bStride = m2.STRIDE;

    const float32_t *  A M_ALIGNED = m1.memory();
    const float32_t *  B M_ALIGNED = m2.memory();
    float32_t *  C M_ALIGNED = (float32_t*)(this->memory());

    int i_row, rows;

    constexpr int C4 = COLUMNS & ~0x03;
    constexpr int R4 = ROWS & ~0x3;
    constexpr int Z4 = Z & ~0x3;

    for (i_row = 0; i_row < R4; i_row += 4)
    {   

        for (int i_col =0; i_col < bCols; ++i_col)
        {
            const float32_t * pB = B + i_col*bStride;

            const float32_t *   pA M_ALIGNED = A + i_row;


            float32x4_t C0 = vmovq_n_f32(0);
            float32x4_t C1 = vmovq_n_f32(0);

            constexpr int Z4 = Z & ~0x03;

            if (Z4 < 16) 
            {
                if (Z4 >= 4)
                {
                    float32x2_t B0 = vld1_f32(pB); 
                    float32x2_t B0a = vld1_f32(pB+2); 
                    pB += 4;

                    float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);

                    float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A1,B0,1);

                    float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                    float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A3,B0a,1);
                }
                if (Z4 >= 8)
                {
                    float32x2_t B0 = vld1_f32(pB); 
                    float32x2_t B0a = vld1_f32(pB+2); 
                    pB += 4;

                    float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);

                    float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A1,B0,1);

                    float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                    float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A3,B0a,1);
                }
                if (Z4 >= 12)
                {
                    float32x2_t B0 = vld1_f32(pB); 
                    float32x2_t B0a = vld1_f32(pB+2); 
                    pB += 4;

                    float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);

                    float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A1,B0,1);

                    float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                    float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A3,B0a,1);
                }


            } else {
                int i_z;
                for (i_z = 0; i_z < Z4; i_z += 4)
                {
                    float32x2_t B0 = vld1_f32(pB); 
                    float32x2_t B0a = vld1_f32(pB+2); 
                    pB += 4;

                    float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A0,B0,0);

                    float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A1,B0,1);

                    float32x4_t A2 = vld1q_f32(pA); pA += aStride;
                    C0 = vmlaq_lane_f32(C0,A2,B0a,0);

                    float32x4_t A3 = vld1q_f32(pA); pA += aStride;
                    C1 = vmlaq_lane_f32(C1,A3,B0a,1);
                }
            }
            if (Z & 2) {
                float32x2_t B0 = vld1_f32(pB);
                pB += 2;

                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);

                float32x4_t A1 = vld1q_f32(pA); pA += aStride;
                C1 = vmlaq_lane_f32(C1,A1,B0,1);
            }
            if (aCols & 1)
            {
                float32x2_t B0 = vmov_n_f32(*pB);
                pB += 1;
                float32x4_t A0 = vld1q_f32(pA); pA += aStride;
                C0 = vmlaq_lane_f32(C0,A0,B0,0);
            }
            C0 = vaddq_f32(C0,C1);
            float32_t*pC = C + i_row + i_col*this->STRIDE;
            vst1q_f32(pC,C0);
        }
    }
    if (aRows & 0x02) {
        for (int i_col =0; i_col < bCols; ++i_col)
        { 
            const float32_t *pB M_ALIGNED = B + i_col*bStride;

            const float32_t *pA  M_ALIGNED = A + i_row;

            float32x2_t A0, A1, A2, A3;

            float32x2_t C0 = vmov_n_f32(0);
            float32x2_t C1 = vmov_n_f32(0);

            for (int i_z = 0; i_z < Z4; i_z += 4)
            {
                float32x2_t B0 = vld1_f32(pB); 
                float32x2_t B0a = vld1_f32(pB+2); 
                pB += 4;

                float32x2_t A0 = vld1_f32(pA); pA += aStride;
                C0 = vmla_lane_f32(C0,A0,B0,0);

                float32x2_t A1 = vld1_f32(pA); pA += aStride;
                C1 = vmla_lane_f32(C1,A1,B0,1);

                float32x2_t A2 = vld1_f32(pA); pA += aStride;
                C0 = vmla_lane_f32(C0,A2,B0a,0);

                float32x2_t A3 = vld1_f32(pA); pA += aStride;
                C1 = vmla_lane_f32(C1,A3,B0a,1);

            }
            if (Z & 2) {
                float32x2_t B0 = vld1_f32(pB);
                pB += 2;

                float32x2_t A0 = vld1_f32(pA); pA += aStride;
                C0 = vmla_lane_f32(C0,A0,B0,0);

                float32x2_t A1 = vld1_f32(pA); pA += aStride;
                C1 = vmla_lane_f32(C1,A1,B0,1);

            }
            if (Z & 1)
            {
                float32x2_t B0 = vld1_f32(pB);
                pB += 1;
                float32x2_t A0 = vld1_f32(pA); pA += aStride;
                C0 = vmla_lane_f32(C0,A0,B0,0);
            }
            C0 = vadd_f32(C0,C1);
            float32_t*pC = C + i_row + i_col*this->STRIDE;
            vst1_f32(pC,C0);
        }

        rows -= 2;
        i_row += 2;
    }
    if (aRows & 0x1) {
        // one last row. use non-vector math.
        for (int i_col =0; i_col < bCols; ++i_col)
        {
            const float32_t *pB = B + i_col*bStride;


            const float32_t *pA = A + i_row;

            float32_t A0, A1, A2, A3;

            float32_t C0 = 0;

            int i_z;
            for (i_z = 0; i_z < aCols; ++i_z)
            {
                C0 += (*pA)*(*pB);
                pB += 1;
                pA += aStride;
            }
            
            float32_t*pC = C + i_row + i_col*this->STRIDE;
            *pC = C0;
        }

        rows -= 2;
        i_row += 1;
    }
}
template <int ROWS,int COLUMNS>
bool Matrix<ROWS,COLUMNS,float32_t>::operator==(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other)
{
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLUMNS; ++c) {
            if ((*this)(r,c) != other(r,c)) 
            return false;
        }
    }
    return true;
}
template <int ROWS,int COLUMNS>
bool Matrix<ROWS,COLUMNS,float32_t>::operator!=(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other)
{
    return !((*this) == other);
}

template<int ROWS, int COLUMNS>
std::string Matrix<ROWS,COLUMNS,float32_t>::toString()
{
    std::stringstream ss;
    for (int r = 0; r < ROWS; ++r)
    {
        ss << "[";
        for (int c = 0; c < COLUMNS; ++c)
        {
            ss << std::setw(8) << (*this)(r,c) << " ";
        }
        ss << ']' << std::endl;
    }    
    return ss.str();
}


#if 0
template<int ROWS,int COLUMNS>
void Matrix<ROWS,COLUMNS,float32_t>::InvertAssign(const Matrix<ROWS,COLUMNS,float_t>&m1)
{
    std::static_assert(ROWS == COLUMNS);
    // gauss-jordan.

    // to improve memory access order, we pretend that the matrices are in row-major format
    // even though they are in column-major format. Because transpose((transponse(A)^1)) = A^1

    this->SetIdentity();

    // use m1 as the source matrix, for the first pass, after which we can use copied data.
    Matrix<ROWS,COLUMNS,float_t> workingMatrix;
    Matrix<ROWS,COLUMNS,float_t> *pSourceMatrix = &m1;
    Matrix<ROWS,COLUMNS,float_t> *pDestMatrix = &workingMatrix;
    bool firstTime = true;

    // reduce to upper triangular format 
    int r, c, r2, c2;
    const int nRows = 8;  // e.g. rows are actually columns from the column-major perspective. &c.
    const int nCols = 8;
    for (int r = 0; r < nRows; ++r)
    {
        float *  pPivot M_ALIGNED= pSourceMatrix->columnPointer(r);

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
            pColValue += SPAN;
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

        constexpr R4 = ROW & ~3;

        if (pivotRow != r)
        {
            // swap the pivot row and normalize it.
            float * pSrc M_ALIGNED = pSourceMatrix->columnPointer(pivotRow);
            float *pDestPivot M_ALIGNED = pDestMatrix->columnPointer(r);
            float *pDestSrc M_ALIGNED = pDestMatrix->columnPointer(pivotRow);

            int i_col;
            if (R4 < 16)
            {
                if (R >= 4)
                {
                    float32x4_t T0 = vld1q_f32(pSrc);
                    float32x4_t S0 = vld1q_f32(pPivot);
                    vst1q_f32(pDestSrc,S0);

                    T0 = vmulq_f32(T0,INV);
                    vst1q_f32(pDestPivot,T0);
                    vst1q_f32(pPivot+4,T1);
                }
                if (R4 >= 8)
                {
                    float32x4_t T0 = vld1q_f32(pSrc+4);
                    float32x4_t S0 = vld1q_f32(pPivot+4);
                    vst1q_f32(pDestSrc+4,S0);

                    T0 = vmulq_f32(T0,INV);
                    vst1q_f32(pDestPivot+4,T0);
                }
                if (R4 >= 12)
                {
                    float32x4_t T0 = vld1q_f32(pSrc+8);
                    float32x4_t S0 = vld1q_f32(pPivot+8);
                    vst1q_f32(pDestSrc+4,S0);

                    T0 = vmulq_f32(T0,INV);
                    vst1q_f32(pDestPivot+8,T0);
                }
                i_col = R4;
            } else {
                for (i_col = 0; i_col < R4; i_col += 4)
                {
                    float32x4_t T0 = vld1q_f32(pSrc+i);
                    float32x4_t S0 = vld1q_f32(pPivot+i);
                    vst1q_f32(pDestSrc+i,S0);

                    T0 = vmulq_f32(T0,INV);
                    vst1q_f32(pDestPivot+i,T0);

                }
            }
            if (R4 & 2)
            {
                float32x2_t T0 = vld1_f32(pSrc+i_col);
                float32x2_t S0 = vld1_f32(pPivot+i_col);
                vst1_f32(pDestSrc+i_col,S0);

                T0 = vmulq_f32(T0,INV);
                vst1q_f32(pDestPivot++i_col,T0);
                i_col += 2;
            }
            if (R4 & 1)
            {
                float32x2_t T0 = vmov_n_f32(*(pSrc+i_col));
                float32x2_t S0 = vmov_n_f32(*(pPivot+i_col));
                vst1_f32(pDestSrc+i_col,S0);

                T0 = vmulq_f32(T0,INV);
                vst1q_f32(pDestPivot++i_col,T0);
            }
            // Fix rounding errors on the diagonal.
            pPivot[r] = 1; // instead of vsetq. Is this a serious stall?


            // now on the aux matrix.
            float *pInvSrc = this->columnPointer(pivotRow);
            float *pInvPivot = this->columnPointer(r);

            if (R4 < 16)
            {
                if (R4 >= 4)
                {
                    float32x4_t TAUX0 = vld1q_f32(pInvSrc);
                    float32x4_t S0 = vld1q_f32(pInvPivot);

                    vst1q_f32(pInvSrc,S0);
                    TAUX0 = vmulq_f32(TAUX0,INV);
                    vst1q_f32(pInvPivot,TAUX0);
                }
                if (R4 >= 8)
                {
                    float32x4_t TAUX0 = vld1q_f32(pInvSrc+4);
                    float32x4_t S0 = vld1q_f32(pInvPivot+4);

                    vst1q_f32(pInvSrc+4,S0);
                    TAUX0 = vmulq_f32(TAUX0,INV);
                    vst1q_f32(pInvPivot+4,TAUX0);
                }
                if (R4 >= 12)
                {
                    float32x4_t TAUX0 = vld1q_f32(pInvSrc+8);
                    float32x4_t S0 = vld1q_f32(pInvPivot+8);

                    vst1q_f32(pInvSrc+8,S0);
                    TAUX0 = vmulq_f32(TAUX0,INV);
                    vst1q_f32(pInvPivot+8,TAUX0);
                }
                i_col = R4;
            } else { // R4 >= 16
                for (i_col = 0; i_col < R4; i_col += 4)
                {
                    float32x4_t TAUX0 = vld1q_f32(pInvSrc+i_col);
                    float32x4_t S0 = vld1q_f32(pInvPivot+i_col);

                    vst1q_f32(pInvSrc+i_col,S0);
                    TAUX0 = vmulq_f32(TAUX0,INV);
                    vst1q_f32(pInvPivot+i_col,TAUX0);
                }
            }
            if (ROWS & 2)
            {
                float32x2_t TAUX0 = vld1_f32(pInvSrc+i_col);
                float32x2_t S0 = vld1_f32(pInvPivot+i_col);

                vst1_f32(pInvSrc+i_col,S0);
                TAUX0 = vmul_f32(TAUX0,INV);
                vst1_f32(pInvPivot+i_col,TAUX0);
                i_col += 2;
            }
            if (ROWS & 1)
            {
                float32x2_t TAUX0 = vmov_n_f32(pInvSrc[i_col]);
                pInvSrc[i_col] = pInvPivot[i_col];
                TAUX0 = vmul_f32(TAUX0,INV);
                pInvPivot[i_col] = vget_lane_f32(TAUX0,0);
                i_col += 1;
            }
        }
        else
        {
            if (R4 < 16)
            {
                if (R4 == 4)
                {
                    float32x4_t T0 = vld1q_f32(pPivot);
                    T0 = vmulq_f32(T0,INV);
                    vlst1q_f32(pPivot,T0);
                }
                if (R4 == 8)
                {
                    float32x4_t T0 = vld1q_f32(pPivot);
                    float32x4_t T1 = vld1q_f32(pPivot+4);
                    T0 = vmulq_f32(T0,INV);
                    T1 = vmulq_f32(T1,INV);
                    vlst1q_f32(pPivot,T0);
                    vlst1q_f32(pPivot+4,T1);
                }
                if (R4 == 12)
                {
                    float32x4_t T0 = vld1q_f32(pPivot);
                    float32x4_t T1 = vld1q_f32(pPivot+4);
                    float32x4_t T2 = vld1q_f32(pPivot+8);
                    T0 = vmulq_f32(T0,INV);
                    T1 = vmulq_f32(T1,INV);
                    T2 = vmulq_f32(T2,INV);
                    vlst1q_f32(pPivot,T0);
                    vlst1q_f32(pPivot+4,T1);
                    vlst1q_f32(pPivot+8,T2);
                }
                i_col = R4;
            } else {
                constexpr int R8 = ROWS & ~0x07;
                for (i_col = 0; i_col < R8; i_col += 8)
                {
                    float32x4_t T0 = vld1q_f32(pPivot+i_col);
                    float32x4_t T1 = vld1q_f32(pPivot+i_col+4);
                    T0 = vmulq_f32(T0,INV);
                    T1 = vmulq_f32(T1,INV);

                    vst1q_f32(pPivot+i_col,T0);
                    vst1q_f32(pPivot+i_col+4,T1);
                }
                if (ROWS & 4) 
                {
                    float32x4_t T0 = vld1q_f32(pPivot+i_col);
                    T0 = vmulq_f32(T0,INV);
                    vst1q_f32(pPivot+i_col,T0);
                    i_col += 4;
                }
            }
            if (ROWS & 2)
            {
                float32x2_t T0 = vld1_f32(pPivot+i_col);
                T0 = vmulq_f32(T0,vlow(INV));
                vst1_f32(pPivot+i_col,T0);
                i_col += 2;
            }
        }
        // T0, T1 contain the pivot row.
        // TAUX0, TAUX1 contain the pivot row in the auxilliary matrix.
    xxx has problem. T0,T1 do NOT contain the pivot row.
        float32_t * ALIGNED_16 pRow = pPivot+8;
        float32_t * ALIGNED_16 pInvRow = inv->colptr(r+1);

        for (int r2 = r + 1; r2 < nRows; ++r2)
        {
            float32_t t0 = pRow[r];

            float32x4_t NEG = vmovq_n_f32(-t0);
            float32x4_t S0 = vld1q_f32(pRow);
            float32x4_t S1 = vld1q_f32(pRow+4);
            S0 = vmlaq_f32(S0,T0,NEG); 
            //S0 = vsetq_lane_f32(0,S0,r); // clear rounding errors. should cancel perfectly.

            S1 = vmlaq_f32(S1,T1,NEG); 
            vst1q_f32(pRow,S0);
            vst1q_f32(pRow+4,S1);
            pRow[r] = 0;

            S0 = vld1q_f32(pInvRow);
            S1 = vld1q_f32(pInvRow+4);
            S0 = vmlaq_f32(S0,TAUX0,NEG);
            S1 = vmlaq_f32(S1,TAUX1,NEG);
            vst1q_f32(pInvRow,S0);
            vst1q_f32(pInvRow+4,S1);

            pRow += 8;
            pInvRow += 8;
        }
    }

    // backsolve the upper triangle.
    for (int r = nRows - 1; r >= 1; --r)
    {
        float * ALIGNED_16 prInvSrc0 = inv->colptr(r);
        float * ALIGNED_16 prInvDest = inv->colptr(0);
        float *prDest = m->colptr(0) + r;

        float32x4_t INVSRC_0 = vld1q_f32(prInvSrc0);
        float32x4_t INVSRC_4 = vld1q_f32(prInvSrc0+4);
        for (int r2 = 0; r2 < r; ++r2)
        {
            float val = *prDest;

            float32x4_t VAL = vmovq_n_f32(-val);

            float32x4_t INVDST_0 = vld1q_f32(prInvDest);
            float32x4_t INVDST_4 = vld1q_f32(prInvDest+4);
            INVDST_0 = vmlaq_f32(INVDST_0,INVSRC_0,VAL);
            INVDST_4 = vmlaq_f32(INVDST_4,INVSRC_4,VAL);
            vst1q_f32(prInvDest,INVDST_0);
            vst1q_f32(prInvDest+4,INVDST_4);

            prDest += nRows;
            prInvDest += nRows;
        }
    }

}

#endif



#endif