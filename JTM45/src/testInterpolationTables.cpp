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
#include "rt-wdf_interpolatedTable.h"
#include <chrono>
#include <iostream>
#include <random>

static double TriodeModelIk(double vAC_, double vGC_)
{
    constexpr double G = 2.242E-3;
    constexpr double C = 3.40;
    constexpr double Mu = 103.2;
    constexpr double Y = 1.26;
    constexpr double InvY = 1/Y;
    constexpr double InvMu = 1/Mu;
    constexpr double InvC = 1/C;

    constexpr double Gg = 6.177E-4;
    constexpr double Cg = 9.901;
    constexpr double InvCg = 1/Cg;
    constexpr double E = 1.314;
    constexpr double InvE = 1/E;
    constexpr double Ig0 = 8.025E-8;

    const double vAC_mu = vAC_ * InvMu;
    const double vGC = vGC_;

    const double exp_C_vAC_mu_vGC = exp( C * ( vAC_mu + vGC ));
    double log_1_exp_C_vAC_mu_vGC_InvC = (log(1 + exp_C_vAC_mu_vGC) * InvC);
    double pow_log_1_exp_C_vAC_mu_vGC_C_y_1 = pow( log_1_exp_C_vAC_mu_vGC_InvC, (Y-1));

        const double exp_Cg_vGC = exp( Cg * vGC );
    const double log_1_exp_Cg_vGC_Cg = (log( 1 + exp_Cg_vGC ) * InvCg);

    double pow_log_1_exp_Cg_vGC_Cg_E_1 = pow( log_1_exp_Cg_vGC_Cg, (E-1));
    // Ig
    //* (*fNL)((*currentPort)+1) = Gg * pow( log_1_exp_Cg_vGC_Cg, E ) + Ig0;
    double Ig = Gg * pow_log_1_exp_Cg_vGC_Cg_E_1*log_1_exp_Cg_vGC_Cg + Ig0;    
    // Ik 
        //* (*fNL)(*currentPort) = G * pow( log( 1 + exp_C_vAC_mu_vGC ) *  InvC , Y ) - (*fNL)((*currentPort)+1);
    return G * pow_log_1_exp_C_vAC_mu_vGC_C_y_1*log_1_exp_C_vAC_mu_vGC_InvC - Ig;
}
static double TriodeModelIg(double vGC_)
{
    constexpr double G = 2.242E-3;
    constexpr double C = 3.40;
    constexpr double Mu = 103.2;
    constexpr double Y = 1.26;
    constexpr double InvY = 1/Y;
    constexpr double InvMu = 1/Mu;
    constexpr double InvC = 1/C;

    constexpr double Gg = 6.177E-4;
    constexpr double Cg = 9.901;
    constexpr double InvCg = 1/Cg;
    constexpr double E = 1.314;
    constexpr double InvE = 1/E;
    constexpr double Ig0 = 8.025E-8;

    const double vGC = vGC_;

    const double exp_Cg_vGC = exp( Cg * vGC );
    const double log_1_exp_Cg_vGC_Cg = (log( 1 + exp_Cg_vGC ) * InvCg);

    double pow_log_1_exp_Cg_vGC_Cg_E_1 = pow( log_1_exp_Cg_vGC_Cg, (E-1));
    // Ig
    //* (*fNL)((*currentPort)+1) = Gg * pow( log_1_exp_Cg_vGC_Cg, E ) + Ig0;
    double Ig = Gg * pow_log_1_exp_Cg_vGC_Cg_E_1*log_1_exp_Cg_vGC_Cg + Ig0;    

    return Ig;
}


static std::default_random_engine randGenerator;
static std::uniform_real_distribution<rt_float> realDistribution(0.0,1.0);

static rt_float random(rt_float min, rt_float max)
{
    return min + (max-min)*realDistribution(randGenerator);
}

void getOriginalResult(rt_float x, rt_float y,rt_float*val, rt_float*dValdX,rt_float*dValDy)
{
    *val = TriodeModelIk(x,y);
    const double h = 0.001;

    *dValdX = (TriodeModelIk(x+h/2,y)-TriodeModelIk(x-h/2,y))/h;
    *dValDy  = (TriodeModelIk(x,y+h/2)-TriodeModelIk(x,y-h/2))/h;


}

void test2dInterpolator(const interpolated2dTable<128,512>& table, rt_float x, rt_float y)
{
    rt_float tableResult, dx,dy;
    table.get(x,y,&tableResult,&dx,&dy);
    rt_float originalResult,originalDx,originalDy;
    getOriginalResult(x,y,&originalResult,&originalDx,&originalDy);


    rt_float error = std::abs(tableResult-originalResult);
    if (error > 1e-6)
    {
        std::cout << "e: " << error << " (" << x << "," << y << ") orig: " << originalResult << " interp: " << tableResult << std::endl;
    }
    rt_float dxError = std::abs(originalDx-dx);
    if (dxError > 1e-5)
    {
        std::cout << "e(dx): " << dxError << " (" << x << "," << y << ")= orig: " << originalDx << " interp: " << dx << std::endl;
    }
    rt_float dyError = std::abs(originalDy-dy);
    if (dyError > 1e-5)
    {
        std::cout << "e(dy): " << dyError << " (" << x << "," << y << ")= orig: " << originalDy << " interp:  " << dy << std::endl;
    }
    REQUIRE(error < 1e-4);
    REQUIRE(dyError < 1e-4);
    REQUIRE(dxError < 1e-4);

}

TEST_CASE( "2D interpolationTable tests", "[interpolationTable2d][interpolationTables]" ) {
    interpolated2dTable<128,512> table(
        5,350,
        -5,5,
        TriodeModelIk
    );

    test2dInterpolator(table,350.0,4.999999);
    test2dInterpolator(table,349.99999,5);


    test2dInterpolator(table,265.684,-0.413498);
    // check boundary conditions.
    test2dInterpolator(table,350,5);
    test2dInterpolator(table,5,5);
    test2dInterpolator(table,5,-5);
    test2dInterpolator(table,350,-5);
    
    // test samples against original
    for (int i = 0; i < 500000; ++i)
    {
        rt_float x = random(5,350);
        rt_float y = random(-5,5);

        test2dInterpolator(table,x,y);
    }

    using Clock=std::chrono::high_resolution_clock;

    constexpr uint64_t ITERATIONS = 100*50000LL;
    Clock::time_point t0 = Clock::now();
    rt_float total = 0;

    for (uint64_t i = 0; i < ITERATIONS; ++i)
    {
        rt_float tableResult, dx,dy;
        rt_float t = i*1E-7;
        table.get(100+t,1+t,&tableResult,&dx,&dy);
    }
    Clock::time_point t1 = Clock::now();
    for (uint64_t i = 0; i < ITERATIONS; ++i)
    {
        rt_float t = i*1E-7;
        total += 1E-12*TriodeModelIk(100+t,-1 + i*1E-7);
    }
    Clock::time_point t2 = Clock::now();


    std::chrono::duration<rt_float> dt = t1-t0;
    std::chrono::duration<rt_float> dtOriginal = t2-t1;
    std::cout << "---- testInterpolationTables --" << std::endl;
    std::cerr << "Interpolation Time: " << dt.count() << std::endl;
    std::cerr << "Original Time: " << dtOriginal.count() << "      " << total << std::endl;

}

void testInterpolated1dTable(interpolatedTable&table, rt_float x)
{

        rt_float tableResult, dx;
        table.get(x,&tableResult,&dx);
        rt_float originalResult = TriodeModelIg(x);
        rt_float h = 0.01;
        rt_float expectedDx = (TriodeModelIg(x+h/2)-TriodeModelIg(x-h/2))/h;

        rt_float error = std::abs(tableResult-originalResult);
        if (error > 1e-7)
        {
            std::cout << "e: " << error << " (" << x << ")= expected: " << originalResult << " actual: " << tableResult << std::endl;
        }
        rt_float errorDx = std::abs(dx-expectedDx);
        if (error > 1e-5)
        {
            std::cout << "eDx: " << errorDx << " (" << x << ")= expected: " << expectedDx << " actual: " << dx << std::endl;
        }
        REQUIRE(error < 1e-5);
        REQUIRE(errorDx < 1e-5);

}

TEST_CASE( "interpolationTable tests", "[interpolationTable1d][interpolationTables]" ) {
    interpolatedTable table(
        1000,-13,13,
        TriodeModelIg
    );
    testInterpolated1dTable(table,-13);
    testInterpolated1dTable(table,13);
    
    for (int i = 0; i < 500000; ++i)
    {
        rt_float y = random(-3,3);
        testInterpolated1dTable(table,y);
    }

    using Clock=std::chrono::high_resolution_clock;

    constexpr uint64_t ITERATIONS = 100*50000LL;
    Clock::time_point t0 = Clock::now();
    rt_float total = 0;

    for (uint64_t i = 0; i < ITERATIONS; ++i)
    {
        rt_float tableResult, dy;
        rt_float t = i*1E-7;
        table.get(-1+t,&tableResult,&dy);
        total += 1e-12*tableResult;
    }
    Clock::time_point t1 = Clock::now();
    for (uint64_t i = 0; i < ITERATIONS; ++i)
    {
        rt_float t = i*1E-7;
        total += 1E-12*TriodeModelIg(-1+t);
    }
    Clock::time_point t2 = Clock::now();


    std::chrono::duration<rt_float> dt = t1-t0;
    std::chrono::duration<rt_float> dtOriginal = t2-t1;
    std::cout << "---- Ig testInterpolationTables --" << std::endl;
    std::cerr << "Interpolation Time: " << dt.count() << std::endl;
    // ---                                                   prevent optimized elmination of the loops(!)
    std::cerr << "Original Time: " << dtOriginal.count() << "        garbage(" << total << ")" << std::endl;

}