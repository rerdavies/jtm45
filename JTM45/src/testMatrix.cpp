// Copyright (c) 2021 Robin Davies
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <catch/catch.hpp>
#include <iostream>

#include "matrix.h"

template<int ROWS, int COLUMNS> void AllocTest()
{
    // make sure that mem and stride were correctly assigned.
    Matrix<ROWS,COLUMNS> m;
    for (int r = 0; r < ROWS; ++r)
    {
        for (int c = 0; c < COLUMNS; ++c)
        {
            m(r,c) = r*100+c;
        }
    }
    Matrix<ROWS,COLUMNS> m2 = m;
    for (int r = 0; r < ROWS; ++r)
    {
        for (int c = 0; c < COLUMNS; ++c)
        {
            assert(m(r,c) == m2(r,c));
        }
    }

}

TEST_CASE( "matrix construction", "[matrixAllocTest][matrix]" ) {
    AllocTest<1,1>();
    AllocTest<2,1>();
    AllocTest<3,2>();
    AllocTest<4,3>();
    AllocTest<5,3>();
}

TEST_CASE( "vector add", "[vectorAdd][matrix]" ) {
    Vector<2> v1({1,1}),v2({2,2});

    Vector<2> v3 = v1+v2;
    REQUIRE(v3 == Vector<2>({3,3}));

    v3 = v2+v2;
    REQUIRE(v3 == Vector<2>({4,4}));

    v3 = v1+v2;

    v3 = v1+v2+v2;
    REQUIRE(v3 == Vector<2>({5,5}));


    v3 = v2-v1+v2;
    REQUIRE(v3 == Vector<2>({3,3}));

    v3 = v2+v2-v1;
    REQUIRE(v3 == Vector<2>({3,3}));

    v3 = -v2;
    REQUIRE(v3 == Vector<2>({-2,-2}));

    v3 = -(v2) + -(v2+v2);
    REQUIRE(v3 == Vector<2>({-6,-6}));

    v3 = -(v2) + (v1-v1) -(v2+v2);
    REQUIRE(v3 == Vector<2>({-6,-6}));

#ifdef ERROR
    v3 = 3; // error

    Vector<3> error(3);
#endif
}

TEST_CASE( "matrix multiply", "[matrixMultiply][matrix]" ) {
    {
        Matrix<2,2> m1({ {1,2},{3,4}});
        Matrix<2,2> m2{ {0,1},{1,0}};

        Matrix<2,2> m3 = m1*m2;
        m3 = m1*m2;
        //std::cout << m3.toString() << std::endl;

        REQUIRE(m3 == (Matrix<2,2>{{2,1},{4,3}}));
    }
    {
        Matrix<2,3> m1({ {1,2,3}
                        ,{4,5,6}});
        Matrix<3,2> m2{ 
                {0,1},
                {1,0},
                {1,1}
                };

        Matrix<2,2> m3 = m1*m2;
        m3 = m1*m2;
        //std::cout << m3.toString() << std::endl;

        REQUIRE(m3 == (Matrix<2,2>{{5,4},{11,10}}));
    }
    {
        Matrix<3,2> m1({ {1,2},
                        {3,4},
                        {5,6}});
        Matrix<2,3> m2{ 
                {0,1,1},
                {1,0,1},
                };

        Matrix<3,3> m3 = m1*m2;
        m3 = m1*m2;
        //std::cout << m3.toString() << std::endl;

        REQUIRE(m3 == (Matrix<3,3>{
                {2,1,3},
                {4,3,7},
                {6,5,11}
                }));
    }
}

