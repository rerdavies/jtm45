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
#include <chrono>
#include <iostream>
#include <random>

#define ARM_NEON
#include <rt-wdf_types.h>
#include <rt-wdf_arm_optimizations.h>

static std::default_random_engine randGenerator(1);
static std::uniform_real_distribution<rt_float> realDistribution(0.0,1.0);

static rt_float random(rt_float min, rt_float max)
{
    return min + (max-min)*realDistribution(randGenerator);
}

static bool areEqual(rt_float v0, rt_float v1)
{
    return std::abs(v0-v1) < 1E-6;
}

static void randomize(Mat<rt_float> &m) {
    for (int r = 0; r < m.n_rows; ++r)
    {
        for (int c = 0; c < m.n_cols; ++c)
        {
            m(r,c) = random(-1,1);
        }
    }
}

static bool areEqual(Mat<float>&m1, Mat<float>&m2) {
    if (m1.n_rows != m2.n_rows || m1.n_cols != m2.n_cols) return false;

    for (int r = 0; r < m1.n_rows; ++r)
    {
        for (int c = 0; c < m1.n_cols; ++c)
        {
            float v1 = m1(r,c);
            float v2 = m2(r,c);
            float e;
            if (std::abs(v1) < 1)
            {
                e = std::abs(v1-v2);
            } else {
                e = std::abs(v1-v2)/v1;
            }
            if (e >= 1E-3)  {
                return false;
            }
        }
    }
    return true;
}



TEST_CASE( "ARM NEON Matrix Inversion", "[armInvert]" ) {
    Mat<float> t{
        {0.3, 1, -0.1},
        {0.25,0,2},
        {1,0,0.01}};
    t.print();
    cout << "=========" << endl;
    Mat<float> invT(3,3);
    Mat<float> t2 = t;
    neon_invert(&t2,&invT);

    Mat<float> m = t*invT;

    t.print();
    cout << endl;
    m.print();

} 

void printMatrix(Mat<float>&matrix)
{
    matrix.print();
}

TEST_CASE( "ARM NEON 8x8 Matrix Inversion", "[armInvert88]" ) {
    Mat<float> t(8,8);
    Mat<float> inverse(8,8);

    for (int i = 0; i < 1000; ++i)
    {
        randomize(t);


        Mat<float> expected = t.i();

        Mat<float> t2 = t;

        neon_invert_88(&t2,&inverse);
        if (!areEqual(inverse,expected)) {
            printMatrix(inverse);
            cout << endl;
            expected.print();
        }
        REQUIRE(areEqual(inverse,expected) == true);

    }

} 