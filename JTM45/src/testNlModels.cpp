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
#include "rt-wdf_nlModels.h"

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

struct  NlNodeResult {
    NlNodeResult()
    : fNL(2u),
      jNL(2u,2u)
      {

      }
    Col<rt_float> fNL;
    Mat<rt_float> jNL;

    bool matches(NlNodeResult &other) {
        return areEqual(fNL(0),other.fNL(0))
        && areEqual(fNL(1),other.fNL(1))
        && areEqual(jNL(0,0),other.jNL(0,0))
        && areEqual(jNL(0,1),other.jNL(0,1))
        && areEqual(jNL(1,0),other.jNL(1,0))
        && areEqual(jNL(1,1),other.jNL(1,1))
        ;
    }

};
std::ostream& operator<<(std::ostream& os, const NlNodeResult& nr)
{
    os << "{(" << nr.fNL(0) << "," << nr.fNL(1) << ") [" << nr.jNL(0,0) << "," << nr.jNL(0,1) << " | " << nr.jNL(1,0) << "," << nr.jNL(1,1) << "]}";
    return os;
}


void getOutput(rt_float x, rt_float y, nlModel &model, NlNodeResult &result)
{
    Col<rt_float> vx(2);

    vx(0) = x;
    vx(1) = y;

    int currentPort = 0;
    model.calculate(&result.fNL,&result.jNL,&vx,&currentPort);


}
static void compareOutputs(rt_float x, rt_float y, nlModel&originalModel, nlModel&newModel)
{
    NlNodeResult originalResult, newResult;
    getOutput(x,y,originalModel,originalResult);
    getOutput(x,y,newModel,newResult);

    if (!originalResult.matches(newResult))
    {
        std::cout << "(" << x << "," << y << ")" << std::endl;
        std::cout << originalResult << std::endl 
                << newResult << std::endl << std::endl;
    } else {
        //std::cout << "OK" << std::endl;
    }
    //REQUIRE(originalResult.matches(newResult) == true);

}

TEST_CASE( "nlModel tests", "[nlModel]" ) {
    triDwModel originalModel;
    interpolatedTriDwModel newModel;


    compareOutputs(298.212,2.93929,originalModel,newModel);
    compareOutputs(260.816,0.194164,originalModel,newModel);

    int currentPort = 0;

    for (int i = 0; i < 500000; ++i)
    {
        currentPort = 0;
        rt_float x = random(-300,300);
        rt_float y = random(-5,5);
        compareOutputs(x,y,originalModel,newModel);

    }


} 