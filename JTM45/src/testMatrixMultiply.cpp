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

using namespace std;

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

static bool areEqual(Mat<float>&m1, Mat<float>&m2) {
    if (m1.n_rows != m2.n_rows || m1.n_cols != m2.n_cols) return false;

    for (int r = 0; r < m1.n_rows; ++r)
    {
        for (int c = 0; c < m1.n_cols; ++c)
        {
            if (std::abs(m1(r,c) -m2(r,c)) >= 1E-6) return false;
        }
    }
    return true;
}

static void testMMult(Mat<float> *result, Mat<float> *m1, Mat<float> *m2)
{
    Mat<float> desired;
    desired = (*m1)*(*m2);

    neon_multiply(result,m1,m2);

    if (!areEqual(desired,*result))
    {
        cout << "==== " 
            << m1->n_cols << "," << m1->n_rows << " x " << m2->n_cols << "," << m2->n_rows << " ===="
            << endl;
        desired.print();
        cout << endl;
        (*result).print();

    }
    REQUIRE( areEqual(desired, *result) == true);

}

static void test4x4() {
    Mat<float> m41{
        {0.3, 1, -0.1,0.2},
        {0.25,0,2,0.2},
        {1,0,0.01,0.2},
        {0.5,2,0.01,0.2}
    };

    Mat<float> m42{
        {1, 0, 0,0},
        {0, 1, 0,0},
        {0,0, 1, 0},
        {0,0,0,1}
    };

    Mat<float> r4(4,4);

    testMMult(&r4,&m41,&m42);

}
static void test3x3() {
    Mat<float> m31{
        {0.3, 1, -0.1},
        {0.25,0,2},
        {1,0,0.01},
    };

    Mat<float> m32{
        {1, 0, 0},
        {0, 1, 0},
        {0,0, 1},
    };

    Mat<float> r3(3,3);

    testMMult(&r3,&m31,&m32);

}

static void setRandom(Mat<float> &m) {
    for (int r = 0; r < m.n_rows; ++r)
    {
        for (int c = 0; c < m.n_cols; ++c)
        {
            m(r,c) = random(-1,1);
        }
    }
}

void testMxNxK(int m, int n, int k)
{
    Mat<float> mMK(m,k,fill::zeros);
    Mat<float> mKN(k,n,fill::zeros);
    Mat<float> result(m,n,fill::zeros);

    setRandom(mMK);
    setRandom(mKN);

    Mat<float> expected = mMK*mKN;
    testMMult(&result,&mMK,&mKN);
}

TEST_CASE( "ARM NEON Matrix Multiply", "[armMultiply]" ) {

    testMxNxK(4,1,3);
    test3x3();
    test4x4();

    for (int m = 1; m < 12; ++m)
    {
        for (int n = 1; n < 12; ++n)
        {
            for (int k = 1; k < 12; ++k) {
                testMxNxK(m,n,k);
            }
        }
    }
}