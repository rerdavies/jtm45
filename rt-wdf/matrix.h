#pragma once
#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>
#include <cassert>
#include <type_traits>
#include <sstream>
#include <iomanip>


#ifndef MATRIX_ALIGNMENT
#define MATRIX_ALIGNMENT 16
#endif

#define M_ALIGNED __attribute__ ((aligned (MATRIX_ALIGNMENT)))



namespace impl {
    /*** Align pointer on the next N-byte boundary */
    template <typename ELEMENT_TYPE>
    ELEMENT_TYPE*alignPointer(ELEMENT_TYPE*p)
    {
        size_t t = (size_t)p;
        t = (t+MATRIX_ALIGNMENT-1)/MATRIX_ALIGNMENT*MATRIX_ALIGNMENT;
        return (ELEMENT_TYPE*)t;
    }
    /*** Pad number of elements of type ELEMENT_TYPE so that it aligns correctly. */
    template <typename ELEMENT_TYPE>
    constexpr size_t padSize(size_t size)
    {
        return (sizeof(ELEMENT_TYPE)*size+MATRIX_ALIGNMENT-1)/MATRIX_ALIGNMENT*MATRIX_ALIGNMENT/sizeof(ELEMENT_TYPE);
    }
};



template <typename DEST_TYPE>
class DelayedOp {
public:
    static constexpr bool IsDelayedOp(const DEST_TYPE &) { return true; }
};

template <int ROWS, int COLUMNS,typename ELEMENT_TYPE> 
class Matrix; 

template <int ROWS, int COLUMNS, typename ELEMENT_TYPE>
class DelayedInvertOp_;

template <int N,typename ELEMENT_TYPE = float_t> class Vector {
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


template <int ROWS, int COLUMNS,typename ELEMENT_TYPE = float_t> 
class Matrix {
public:
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

    void InvertAssign(const MatrixT&m1)
    {
        assert(false); // not implemented.
    }

    DelayedInvertOp_<ROWS,COLUMNS,ELEMENT_TYPE> Invert();

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
//=========== Vector implmentation =================/

template <int N,typename ELEMENT_TYPE>
class DelayedVectorOp: public DelayedOp<Vector<N,ELEMENT_TYPE> > {
public:
    static constexpr int Size = N;
    using ElementTypeT = ELEMENT_TYPE;
    using VectorT = Vector<N,ELEMENT_TYPE>;
};


template <int N, typename ELEMENT_TYPE>
constexpr Vector<N,ELEMENT_TYPE>&Vector<N,ELEMENT_TYPE>::operator=(Vector<N,ELEMENT_TYPE>&v)
{
    for (int i = 0; i < N; ++i)
    {
        this->mem[i] = v[i];
    }
    return *this;
}


template <int N, typename ELEMENT_TYPE>
void Vector<N,ELEMENT_TYPE>::AddAssign(const VectorT&v1, const VectorT&v2)
{
    for (int i = 0; i < N; ++i)
    {
        (*this)[i] = v1[i] + v2[i];
    }
}

template <int N, typename ELEMENT_TYPE>
void Vector<N,ELEMENT_TYPE>::SubtractAssign(const VectorT&v1, const VectorT&v2)
{
    for (int i = 0; i < N; ++i)
    {
        (*this)[i] = v1[i] - v2[i];
    }
}
template <int N, typename ELEMENT_TYPE>
void Vector<N,ELEMENT_TYPE>::NegateAssign(const VectorT&v1)
{
    for (int i = 0; i < N; ++i)
    {
        (*this)[i] = -v1[i];
    }
}
template<int N,typename ELEMENT_TYPE>
template<int COLUMNS>
void Vector<N,ELEMENT_TYPE>::MultiplyAssign(const Matrix<N,COLUMNS,ELEMENT_TYPE> &m,const Vector<COLUMNS,ELEMENT_TYPE> &v)
{
    for (int r = 0; r < N; ++r)
    {
        ELEMENT_TYPE sum = 0;
        for (int c = 0; c < COLUMNS; ++c)
        {
            sum += m(r,c)**v(c);
        }
        (*this)(r) = sum;
    }    
}
template <int N, typename ELEMENT_TYPE>
void Vector<N,ELEMENT_TYPE>::Set(ELEMENT_TYPE value)
{
    for (int i = 0; i < this->Size; ++i) mem[i] = value;
}
template <int N, typename ELEMENT_TYPE>
bool Vector<N,ELEMENT_TYPE>::operator==(const VectorT &other)
{
    for (int i = 0; i < Size; ++i)
    {
        if (mem[i] != other[i]) return false;
    }
    return true;
}
template <int N, typename ELEMENT_TYPE>
bool Vector<N,ELEMENT_TYPE>::operator!=(const VectorT &other)
{
    return !(*this == other);
}

template<int ROWS, int COLUMNS,typename ELEMENT_TYPE>
class DelayedMatVecMulOp_: public DelayedVectorOp<COLUMNS, ELEMENT_TYPE>
{
public:
    using LhsT = Matrix<ROWS,COLUMNS,ELEMENT_TYPE>;
    using RhsT = Vector<COLUMNS,ELEMENT_TYPE>;
    using VectorT = Vector<COLUMNS,ELEMENT_TYPE>;
private:

    const LhsT&lhs;
    const RhsT&rhs;
public:
    DelayedMatVecMulOp_(const LhsT &lhs, const RhsT &rhs)
    :lhs(lhs), rhs(rhs)
    {

    }
    void Apply(VectorT &destination) const {
        destination.MultiplyAssign(lhs,rhs);
    }
    VectorT Apply()
    {
        VectorT result;
        Apply(result);
        return result;
    }
};

template<int ROWS, int COLUMNS,typename ELEMENT_TYPE>
DelayedMatVecMulOp_<ROWS,COLUMNS,ELEMENT_TYPE> operator*(const Matrix<ROWS,COLUMNS, ELEMENT_TYPE> matrix, const Vector<ROWS, ELEMENT_TYPE>& vector)
{
    return DelayedMatVecMulOp_<ROWS,COLUMNS,ELEMENT_TYPE>(matrix,vector);
}





template <int N,typename ELEMENT_TYPE>
class DelayedVectorAdd_: public DelayedVectorOp<N,ELEMENT_TYPE> {
public:
    using VectorT = Vector<N,ELEMENT_TYPE>;
    using LHS = const VectorT&;
    using RHS = const VectorT&;
private:
    LHS v1;
    RHS v2;
public:
    DelayedVectorAdd_(LHS v1, RHS v2)
    :v1(v1),v2(v2)
    {
        
    }
    void Apply(VectorT &dest) const
    {
        dest.AddAssign(v1,v2);
    }

    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }

};
template <int N,typename ELEMENT_TYPE>
class DelayedVectorSubtract_: public DelayedVectorOp<N,ELEMENT_TYPE> {
public:

    using VectorT = Vector<N,ELEMENT_TYPE>;
    using LHS = const VectorT&;
    using RHS = const VectorT&;
private:
    LHS v1;
    RHS v2;
public:
    DelayedVectorSubtract_(LHS v1, RHS v2)
    :v1(v1),v2(v2)
    {
        
    }
    void Apply(VectorT &dest) const
    {
        dest.SubtractAssign(v1,v2);
    }

    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }

};

template <int N,typename ELEMENT_TYPE,  typename LHS,
    typename IS_DELAYED_OP=decltype(LHS::IsDelayedOp(Vector<N,ELEMENT_TYPE>()))
> 
class DelayedVectorAdd2_: public DelayedVectorOp<N,ELEMENT_TYPE> {
private:

    const LHS &lhs;
    const Vector<N,ELEMENT_TYPE>&rhs;
public:
    DelayedVectorAdd2_(const LHS &lhs,const Vector<N,ELEMENT_TYPE> &rhs)
    : lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Vector<N,ELEMENT_TYPE>&destination) const {
        lhs.Apply(destination);
        destination.AddAssign(destination,rhs);
    }
    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};

template <typename LHS,typename RHS,
    typename IS_DELAYED_OP=decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>())),
    typename IS_DELAYED_OP2=decltype(RHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>()))
> 
class DelayedVectorAdd3_: public DelayedVectorOp<LHS::Size,typename LHS::ElementTypeT> {
public:
    constexpr static int N = LHS::Size;
    using ELEMENT_TYPE = typename LHS::ElementTypeT;
private:

    const LHS &lhs;
    const RHS &rhs;
public:
    DelayedVectorAdd3_(const LHS &lhs,const RHS &rhs)
    : lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Vector<N,ELEMENT_TYPE>&destination) const {
        Vector<N,ELEMENT_TYPE> rhs;
        this->rhs.Apply(rhs);

        lhs.Apply(destination);
        destination.AddAssign(destination,rhs);
    }
    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};

template <typename LHS,typename RHS,
    typename IS_DELAYED_OP=decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>())),
    typename IS_DELAYED_OP2=decltype(RHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>()))
> 
class DelayedVectorSubtract3_: public DelayedVectorOp<LHS::Size,typename LHS::ElementTypeT> {
public:
    constexpr static int N = LHS::Size;
    using ELEMENT_TYPE = typename LHS::ElementTypeT;
private:

    const LHS &lhs;
    const RHS &rhs;
public:
    DelayedVectorSubtract3_(const LHS &lhs,const RHS &rhs)
    : lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Vector<N,ELEMENT_TYPE>&destination) const {
        Vector<N,ELEMENT_TYPE> rhs;
        this->rhs.Apply(rhs);

        lhs.Apply(destination);
        destination.SubtractAssign(destination,rhs);
    }
    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};

template <int N,typename ELEMENT_TYPE,  typename LHS,
    typename IS_DELAYED_OP=decltype(LHS::IsDelayedOp(Vector<N,ELEMENT_TYPE>()))
> 
class DelayedVectorSubtract2_: public DelayedVectorOp<N,ELEMENT_TYPE> {
private:

    const LHS &lhs;
    const Vector<N,ELEMENT_TYPE>&rhs;
public:
    DelayedVectorSubtract2_(const LHS &lhs,const Vector<N,ELEMENT_TYPE> &rhs)
    : lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Vector<N,ELEMENT_TYPE>&destination) const {
        lhs.Apply(destination);
        destination.SubtractAssign(destination,rhs);
    }
    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};

template <typename ELEMENT_TYPE, int N, typename LHS,
    typename =decltype(LHS::IsDelayedOp(Vector<N,ELEMENT_TYPE>()))
>
DelayedVectorAdd2_<N,ELEMENT_TYPE,LHS> operator+(const LHS&lhs,const Vector<N,ELEMENT_TYPE>&rhs)
{
    return DelayedVectorAdd2_<N,ELEMENT_TYPE,LHS>(lhs,rhs);
}

template <typename LHS,typename RHS,
    typename =decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>())),
    typename =decltype(LHS::IsDelayedOp(Vector<RHS::Size,typename RHS::ElementTypeT>()))
>
DelayedVectorAdd3_<LHS,RHS> operator+(const LHS&lhs,const RHS&rhs)
{

    return DelayedVectorAdd3_<LHS,RHS>(lhs,rhs);
}
template <typename LHS,typename RHS,
    typename =decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>())),
    typename =decltype(LHS::IsDelayedOp(Vector<RHS::Size,typename RHS::ElementTypeT>()))
>
DelayedVectorSubtract3_<LHS,RHS> operator-(const LHS&lhs,const RHS&rhs)
{

    return DelayedVectorSubtract3_<LHS,RHS>(lhs,rhs);
}


template <typename ELEMENT_TYPE, int N>
DelayedVectorAdd_<N,ELEMENT_TYPE> operator+(const Vector<N,ELEMENT_TYPE>&v1, const Vector<N,ELEMENT_TYPE>&v2)
{
    return DelayedVectorAdd_<N,ELEMENT_TYPE>(v1,v2);
}

template <typename ELEMENT_TYPE, int N, typename LHS,
    typename =decltype(LHS::IsDelayedOp(Vector<N,ELEMENT_TYPE>()))
>
DelayedVectorSubtract2_<N,ELEMENT_TYPE,LHS> operator-(const LHS&lhs,const Vector<N,ELEMENT_TYPE>&rhs)
{
    return DelayedVectorSubtract2_<N,ELEMENT_TYPE,LHS>(lhs,rhs);
}


template <typename ELEMENT_TYPE, int N>
DelayedVectorSubtract_<N,ELEMENT_TYPE> operator-(const Vector<N,ELEMENT_TYPE>&v1, const Vector<N,ELEMENT_TYPE>&v2)
{
    return DelayedVectorSubtract_<N,ELEMENT_TYPE>(v1,v2);
}



template <int N,typename ELEMENT_TYPE>
class DelayedVectorNegate_: public DelayedVectorOp<N,ELEMENT_TYPE> {
public:

    using VectorT = Vector<N,ELEMENT_TYPE>;
    using RHS = const VectorT&;
private:
    RHS rhs;
public:
    DelayedVectorNegate_(RHS rhs)
    : rhs(rhs)
    {
        
    }
    void Apply(VectorT &dest) const
    {
        dest.NegateAssign(rhs);
    }

    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};


template <typename LHS,
    typename IS_DELAYED_OP=decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>()))
> 
class DelayedVectorNegate2_: public DelayedVectorOp<LHS::Size,typename LHS::ElementTypeT> {
private:
    const LHS &value;
    static constexpr int N = LHS::Size;
    using ELEMENT_TYPE = typename LHS::ElementTypeT;
public:
    DelayedVectorNegate2_(const LHS &value)
    : value(value)
    {
    }

    void Apply(Vector<N,ELEMENT_TYPE>&destination) const {
        value.Apply(destination);
        destination.NegateAssign(destination);
    }
    Vector<N,ELEMENT_TYPE> Apply() const
    {
        Vector<N,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};


template <typename LHS,
    typename =decltype(LHS::IsDelayedOp(Vector<LHS::Size,typename LHS::ElementTypeT>()))
>
DelayedVectorNegate2_<LHS> operator-(const LHS&lhs)
{
    return DelayedVectorNegate2_<LHS>(lhs);
}

template <typename ELEMENT_TYPE, int N>
DelayedVectorNegate_<N,ELEMENT_TYPE> operator-(const Vector<N,ELEMENT_TYPE>&v1)
{
    return DelayedVectorNegate_<N,ELEMENT_TYPE>(v1);
}




//=========== Matrix implementation =================/

template <int ROWS, int COLUMNS,typename ELEMENT_TYPE> 
Matrix<ROWS,COLUMNS,ELEMENT_TYPE>&Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::operator=(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE>&other)
{
    int n = STRIDE*Rows;
    for (int i = 0; i < n; ++i)
    {
        this->mem[i] = other.mem[i];
    }
    return *this;
}



template <int ROWS, int COLUMNS,typename ELEMENT_TYPE>
void Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::AddAssign(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m1, const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m2)
{
    for (int r = 0; r < ROWS; ++r)
    {
        ELEMENT_TYPE*pDest = this->columnAddress(r);
        ELEMENT_TYPE*pM1 = m1.columnAddress(r);
        ELEMENT_TYPE*pM2 = m2.columnAddress(r);
        for (int c = 0; c < COLUMNS; ++c)
        {
            *pDest++ = *pM1++ + *pM2++;
        }
    }
}

template <int ROWS, int COLUMNS,typename ELEMENT_TYPE>
void Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::SubtractAssign(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m1, const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m2)
{
    for (int r = 0; r < ROWS; ++r)
    {
        ELEMENT_TYPE*pDest = this->columnAddress(r);
        ELEMENT_TYPE*pM1 = m1.columnAddress(r);
        ELEMENT_TYPE*pM2 = m2.columnAddress(r);
        for (int c = 0; c < COLUMNS; ++c)
        {
            *pDest++ = *pM1++ - *pM2++;
        }
    }

}

template <int ROWS, int COLUMNS,typename ELEMENT_TYPE>
template <int Z>
void Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::MultiplyAssign(const Matrix<Rows,Z,ELEMENT_TYPE> &m1, const Matrix<Z,Columns,ELEMENT_TYPE> &m2)
{
    assert((void*)this != (void*)&m1 && (void*)this != (void*)&m2); // cannot alias matrices when multiplying.

    for (int c = 0; c < COLUMNS; ++c)
    {
        ELEMENT_TYPE *pDest = this->columnAddress(c);
        constexpr int M1_STRIDE = m1.STRIDE;
        for (int r = 0; r < ROWS; ++r)
        {
            ELEMENT_TYPE sum = 0;
            const ELEMENT_TYPE*pM1 = m1.memory()+r;
            const ELEMENT_TYPE*pM2 = m2.columnAddress(c);
            for (int z = 0; z < Z; ++z)
            {
                sum += (*pM1) * (*pM2);
                pM1 += M1_STRIDE;
                ++pM2;
            }
            *pDest++ = sum;
        }
    }
}
template <int ROWS,int COLUMNS, typename ELEMENT_TYPE>
bool Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::operator==(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other)
{
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLUMNS; ++c) {
            if ((*this)(r,c) != other(r,c)) return false;
        }
    }
    return true;
}
template <int ROWS,int COLUMNS, typename ELEMENT_TYPE>
bool Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::operator!=(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &other)
{
    return !((*this) == other);
}

template <int ROWS,int COLUMNS,typename ELEMENT_TYPE>
class DelayedMatrixOp: public DelayedOp<Matrix<ROWS,COLUMNS,ELEMENT_TYPE> > {
public:
    static constexpr int Rows = ROWS;
    static constexpr int Columns = COLUMNS;
    using ElementTypeT = ELEMENT_TYPE;

    using DestinationT = Matrix<ROWS,COLUMNS,ELEMENT_TYPE>;
};

template <int ROWS, int COLUMNS, int Z, typename ELEMENT_TYPE>
class DelayedMatrixMultiplyOp_: public DelayedMatrixOp<ROWS,COLUMNS,ELEMENT_TYPE>
{
public:
    using LhsT = Matrix<ROWS,Z,ELEMENT_TYPE>;
    using RhsT = Matrix<Z,COLUMNS,ELEMENT_TYPE>;
private:
    const LhsT& lhs;
    const RhsT& rhs;
public:
    DelayedMatrixMultiplyOp_(const LhsT&lhs, const RhsT&rhs)
    :lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &destination) const
    {
        destination.MultiplyAssign(lhs,rhs);
    }
    Matrix<ROWS,COLUMNS,ELEMENT_TYPE> Apply() const
    {
        Matrix<ROWS,COLUMNS,ELEMENT_TYPE> result;
        Apply(result);
        return result;
    }
};
template <typename LHS,int ROWS, int COLUMNS,typename ELEMENT_TYPE
    >
class DelayedMatrixMultiplyOp2_: public DelayedMatrixOp<LHS::Rows,COLUMNS,ELEMENT_TYPE>
{
public:
    using LhsT = LHS;
    using RhsT = Matrix<ROWS,COLUMNS,ELEMENT_TYPE>;
private:
    const LhsT& lhs;
    const RhsT& rhs;
public:
    DelayedMatrixMultiplyOp2_(const LhsT&lhs, const RhsT&rhs)
    :lhs(lhs),rhs(rhs)
    {
    }

    void Apply(Matrix<LHS::Rows,COLUMNS,ELEMENT_TYPE> &destination) const
    {
        Matrix<LhsT::Rows,LhsT::Columns> temp;
        lhs.Apply(temp);
        destination.MultiplyAssign(temp,rhs);
    }
    Matrix<LHS::Rows,COLUMNS,ELEMENT_TYPE> Apply() const
    {
        Matrix<LHS::Rows,COLUMNS> result;
        Apply(result);
        return result;
    }
};
template <typename LHS,typename RHS
    >
class DelayedMatrixMultiplyOp3_: public DelayedMatrixOp<LHS::Rows,RHS::COLUMNS,typename LHS::ELEMENT_TYPE>
{
public:
    using LhsT = LHS;
    using RhsT = RHS;
    using DestinationT = Matrix<LHS::Rows,RHS::Columns,typename LHS::ELEMENT_TYPE>;
private:
    const LhsT& lhs;
    const RhsT& rhs;
public:
    DelayedMatrixMultiplyOp3_(const LhsT&lhs, const RhsT&rhs)
    :lhs(lhs),rhs(rhs)
    {
    }

    void Apply(DestinationT &destination) const
    {
        Matrix<LhsT::Rows,LhsT::Columns> tLeft;
        lhs.Apply(tLeft);
        Matrix<RhsT::Rows,RhsT::Columns> tRight;
        rhs.Apply(tRight);
        destination.MultiplyAssign(tLeft,tRight);
    }
    DestinationT Apply() const
    {
        DestinationT result;
        Apply(result);
        return result;
    }
};

template <typename LHS, int ROWS, int COLUMNS, typename ELEMENT_TYPE,
    typename =decltype(LHS::IsDelayedOp(Matrix<LHS::Rows,ROWS,ELEMENT_TYPE>()))
>
DelayedMatrixMultiplyOp2_<LHS,ROWS,COLUMNS,ELEMENT_TYPE> operator*(const LHS&lhs,const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &rhs)
{
    return DelayedMatrixMultiplyOp2_<LHS,ROWS,COLUMNS,ELEMENT_TYPE>(lhs,rhs);
}

template<int ROWS, int COLUMNS, int Z, typename ELEMENT_TYPE>
DelayedMatrixMultiplyOp_<ROWS,COLUMNS,Z,ELEMENT_TYPE>
operator*(const Matrix<ROWS,Z,ELEMENT_TYPE> &lhs, const Matrix<Z,COLUMNS,ELEMENT_TYPE> &rhs)
{
    return DelayedMatrixMultiplyOp_<ROWS,COLUMNS,Z,ELEMENT_TYPE>(lhs,rhs);
}

template<int ROWS, int COLUMNS, typename ELEMENT_TYPE>
std::string Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::toString()
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

template <int ROWS, int COLUMNS, typename ELEMENT_TYPE> 
class DelayedInvertOp_ : DelayedMatrixOp<ROWS,COLUMNS,ELEMENT_TYPE>
{
public:
    typedef Matrix<ROWS,COLUMNS,ELEMENT_TYPE> MatrixT;
private:
    const MatrixT& m;
public:
    DelayedInvertOp_(const Matrix<ROWS,COLUMNS,ELEMENT_TYPE> &m)
    :m(m)
    {
    }

    void Apply(const MatrixT& destination)
    {
        destination.InvertAssign(m);
    }

};

template<int ROWS, int COLUMNS, typename ELEMENT_TYPE>
DelayedInvertOp_<ROWS,COLUMNS,ELEMENT_TYPE> Matrix<ROWS,COLUMNS,ELEMENT_TYPE>::Invert()
{
    return DelayedInvertOp_<ROWS,COLUMNS,ELEMENT_TYPE>(*this);
}


// include specializations for ARM neon.

#include "matrix_arm_neon.h"


#endif
