/*
==============================================================================


MIT LICENSE

Copyright 2021 Robin Davies. rerdavies@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished 
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.

==============================================================================
*/
#ifndef RTWDF_INTERPOLATEDTABLES_H_INCLUDED
#define RTWDF_INTERPOLATEDTABLES_H_INCLUDED

#include <functional>
#include <cassert>

#include "rt-wdf_types.h"
#include <arm_neon.h>


//------------------------------------------------------

/**
 * @brief 2D Table-drive interpolator using Lagrange interpolators.
 * 
 */
template <int xSize,int ySize>
class interpolated2dTable {
private:
    static constexpr int pitch = ((ySize+4)*4);

    rt_float xMin, xM, xMax;
    rt_float yMin, yM, yMax;

    rt_float *pData;

    rt_float* getAt(int ix, int iy) ALIGNED_16
    {
        assert(ix >= 0 && ix < xSize+1);
        assert(iy >= -1 && iy < ySize+3);
        return &(pData[(iy+1)*4 + ix*pitch]);
    }
    const rt_float* getAt(int ix, int iy) const ALIGNED_16 
    {
        assert(ix >= 0 && ix < xSize+1);
        assert(iy >= -1 && iy < ySize+3);
        return &(pData[(iy+1)*4 + ix*pitch]);
    }

    void setValue(int ix, int iy, rt_float a3, rt_float a2, rt_float a1, rt_float a0)
    {
        rt_float *p = getAt(ix,iy);
        p[0] = a0; p[1] = a1; p[2] = a2; p[3] = a3;
    }
    void storeCoefficients(int ix, int iy,std::function<double (double x, double y)>& fn);

    void toXIndex(float_t x,int *ix, float_t *xFrac) const
    {
        float_t t = (x-xMin)*xM;
        float_t tFrac = t-std::floor(t);
        *xFrac = tFrac;
        *ix = (int)(t-tFrac);

    }
    void toYIndex(float_t y, int *iy, float_t*yFrac) const
    {
        float_t t = (y-yMin)*yM;
        float_t tFrac = t-std::floor(t);
        *yFrac = tFrac;
        *iy = (int)(t-tFrac);
    }
public: 
    interpolated2dTable(
        rt_float minX, rt_float maxX,
        rt_float minY, rt_float maxY,
        std::function<double (double x, double y)> function
    );

    ~interpolated2dTable();


    void  get(rt_float x, rt_float y,rt_float *z,rt_float *dZdX, rt_float *dZdY) const;

};

//------------------------------------------------------

/**
 * @brief 1D Table-drive interpolator using Lagrange interpolators approximating a supplied function y=f(x)
 * 
 */


class interpolatedTable {
private:
    int xSize;
    rt_float xMin, xM, xMax, inverseXM;

    rt_float *pData;

    rt_float* getAt(int ix) ALIGNED_16
    {
        assert(ix >= 0 && ix <= xSize);
        return &(pData[ix*4]);
    }

    const rt_float* getAt(int ix ) const ALIGNED_16
    {
        assert(ix >= 0 && ix <= xSize);
        return &pData[ix*4];
    }
    void setValue(int ix, rt_float a0, float a1, float a2, float a3)
    {
        rt_float*p = getAt(ix);
        p[0] = a0;
        p[1] = a1;
        p[2] = a2;
        p[3] = a3;
    }

public: 
    interpolatedTable(
        int xSize, rt_float minX, rt_float maxX,
        std::function<double (double x)> function
    )
        :   xMin(minX),xMax(maxX),xSize(xSize)
    {
        xM = xSize/(maxX-minX);
        inverseXM = 1/xM;
        pData = alignedAlloc<float>((xSize+1)*4);
        rt_float dx = (maxX-minX)/xSize;
        for (int ix = 0; ix <= xSize; ++ix)
        {
            // create lagrange interpolation polynomial
            double x0 = minX + dx*(ix-1);
            double x1 = minX + dx*(ix);
            double x2 = minX + dx*(ix+1);
            double x3 = minX + dx*(ix+2);
            double y0 = function(x0);
            double y1 = function(x1);
            double y2 = function(x2);
            double y3 = function(x3);

            // lagrange interpolation x weights.
            // rt_float c0 = y0*    -(1.0f/6.0f)*x3+(1.0f/2.0f)*x2 - (1.0f/3.0f)*x;
            // rt_float c1 = y1*   (1.0f/2.0f)*x3 -x2 - (1.0f/2.0f)*x+1;
            // rt_float c2 = y2*   -(1.0f/2.0f)*x3 + (1.0f/2.0f)*x2 + x;
            // rt_float c3 = y3 *  (1.0f/6.0f)*x3-(1.0f/6.0f)*x;
            rt_float a0 = float(y1);
            rt_float a1 = float(-(1.0/3.0)*y0 - (1.0/2.0)*y1 + 1*y2 + -1.0/6.0*y3);
            rt_float a2 = float(1.0/2.0*y0 - y1 + 1.0/2.0*y2);
            rt_float a3 = float(-1.0/6.0*y0 + 1.0/2.0*y1 - 1.0/2.0*y2 + 1.0/6.0*y3);
            setValue(ix,a0,a1,a2,a3);
        }
    }

    ~interpolatedTable() { free(pData);}

    //----------------------------------------------------------------------
    /**
     Get the interpolated value of y=f(x), and the slope  f'(x)

     @param x             X value

    @param y              Output Y value, y = f(x)

    @param dYdX           Output slope,  F'(x)

     
    */ 

    void  get(rt_float x_, rt_float *y,rt_float *dYdX) {
        rt_float x = (x_-xMin)*xM;
        int xIndex = (int)std::floor(x);
        x -= xIndex;
        assert(xIndex >= 0 && xIndex <= xSize);

        rt_float *p = getAt(xIndex);

        rt_float x2 = x*x;
        rt_float x3 = x2*x;
        *y = p[3]*x3 + p[2]*x2 + p[1]*x + p[0];
        *dYdX = (3*x2*p[3] + 2*x*p[2] + p[1])*xM;
    }

};



// -------------------- interpolated2dTable inlines. --------
template <int xSize, int ySize>
void interpolated2dTable<xSize,ySize>::storeCoefficients(int ix, int iy,std::function<double (double x, double y)>& fn)
{
    double dx = (xMax-xMin)/(xSize);
    double dy = (yMax-yMin)/(ySize);

    double x0 = (ix-1)*dx + xMin;
    double x1 = (ix)*dx + xMin;
    double x2 = (ix+1)*dx + xMin;
    double x3 = (ix+2)*dx + xMin;

    double y = (iy)*dy + yMin;

    double y0 = fn(x0,y);
    double y1 = fn(x1,y);
    double y2 = fn(x2,y);
    double y3 = fn(x3,y);

    // rt_float c0 = -(1.0f/6.0f)*x3+(1.0f/2.0f)*x2 - (1.0f/3.0f)*x;
    // rt_float c1 = (1.0f/2.0f)*x3 -x2 - (1.0f/2.0f)*x+1;
    // rt_float c2 = -(1.0f/2.0f)*x3 + (1.0f/2.0f)*x2 + x;
    // rt_float c3 = (1.0f/6.0f)*x3-(1.0f/6.0f)*x;

    rt_float a3 = rt_float(y0*-(1.0/6.0) + y1*(1.0/2.0) + y2*(-1.0/2.0) + y3*(1.0/6.0));
    rt_float a2 = rt_float(y0*(1.0/2.0) + y1*-1  + y2*(1.0/2.0) + y3*0);
    rt_float a1 = rt_float(y0*-(1.0/3.0) + y1*(-1.0/2.0)  + y2*(1.0) + y3*(-1.0/6.0));
    rt_float a0 = y1*1;
    this->setValue(ix,iy,a3,a2,a1,a0);
}
template <int xSize, int ySize>
interpolated2dTable<xSize,ySize>::interpolated2dTable(
    rt_float minX, rt_float maxX,
    rt_float minY, rt_float maxY,
    std::function<double (double x, double y)> fn
)
:   xMin(minX),
    yMin(minY),
    xMax(maxX),
    yMax(maxY),
    xM((xSize)/(maxX-minX)),
    yM((ySize)/(maxY-minY))
{

    pData = alignedAlloc<float>(pitch*(xSize+1));
    rt_float xRange = maxX-minX;
    rt_float yRange = maxY-minY;

    for (int iy = -1; iy < ySize+3; ++iy)  
    {
        for (int ix = 0; ix < xSize+1; ++ix)   
        {
            storeCoefficients(ix,iy,fn);
        }
    }
}

template <int xSize, int ySize>
interpolated2dTable<xSize,ySize>::~interpolated2dTable() {
    free(pData);
}

template <int xSize, int ySize>
void  interpolated2dTable<xSize,ySize>::get(rt_float x_, rt_float y_,rt_float *value,rt_float *dDx, rt_float *dDy) const
{
    // prof ~56 multiplies.
    
    int xIndex,yIndex;
    rt_float x,y;
    toXIndex(x_,&xIndex,&x);
    toYIndex(y_,&yIndex,&y);


    {

        float_t x2 = x*x;
        float x3 = x2*x;
//1
        const float_t *pA0 ALIGNED_16 = getAt(xIndex,yIndex);

        float32x4x4_t QMAT = vld4q_f32(pA0-4);
//5
        float32x4_t XVALUES;
        {

            float32x4_t X1 = vmulq_n_f32(QMAT.val[1],x);
            float32x4_t X2 = vmulq_n_f32(QMAT.val[2],x2);
            float32x4_t X3 = vmulq_n_f32(QMAT.val[3],x3);
            float32x4_t X0 = vaddq_f32(QMAT.val[0],X1);
            X2 = vaddq_f32(X2,X3);
            XVALUES = vaddq_f32(X0,X2);
        }

//6
        float y2 = y*y;
        float y3 = y2*y;
//7

        // rt_float c0 = -(1.0f/6.0f)*x3+(1.0f/2.0f)*x2 - (1.0f/3.0f)*x;
        // rt_float c1 = (1.0f/2.0f)*x3 -x2 - (1.0f/2.0f)*x+1;
        // rt_float c2 = -(1.0f/2.0f)*x3 + (1.0f/2.0f)*x2 + x;
        // rt_float c3 = (1.0f/6.0f)*x3-(1.0f/6.0f)*x;

        static float32_t  cx3[4] ALIGNED_16 = {  -(1.0/6.0), (1.0f/2.0f), -1.0f/2.0f, 1.0f/6.0f};
        static float32_t cx2[4] ALIGNED_16 = { 1.0f/2.0f,  -1,  1.0/2.0f,  0 };
        static float32_t cx1[4] ALIGNED_16 = { -1.0f/3.0f, -1.0f/2.0f, 1, -1.0f/6.0f};
        static float32_t cx0[4] ALIGNED_16 = { 0, 1,  0,    0 };


        float32x4_t C0 = vld1q_f32(cx0);
        float32x4_t C1 = vld1q_f32(cx1);
        float32x4_t C2 = vld1q_f32(cx2);
        float32x4_t C3 = vld1q_f32(cx3);
//11
        //*value = lagrangeInterpolate(C0,C1,C2,C3,XVALUES,1,y,y2,y3);
        {
            float32x4_t TC1 = vmulq_n_f32(C1,y);
            float32x4_t TC0 = vaddq_f32(C0,TC1);

            float32x4_t TC2 = vmulq_n_f32(C2,y2);
            float32x4_t TC3 = vmulq_n_f32(C3,y3);
            TC2 = vaddq_f32(TC2,TC3);

            TC0 = vaddq_f32(TC0,TC2);
            float32x4_t V = vmulq_f32(TC0,XVALUES);
            float32x2_t VW = vadd_f32(vget_low_f32(V),vget_high_f32(V));
            *value =  vget_lane_f32(vpadd_f32(VW,VW),0);
        }

        float32_t dy2 = 2*y;
        float32_t dy3 = 3*y2;
//12
        // d/dy of a lagrange interpolation of the Y[0..n] interpolations.
        // evaluate with x^n terms replaced with their 
        // derivatives: 0, 1, 2*x, 3*x^3

        //*dDy = lagrangeInterpolate(C0,C1,C2,C3,XVALUES,0,1,2*y,3*y*y)*this->yM;
        {
            float32x4_t TC2 = vmulq_n_f32(C2,dy2);
            float32x4_t TC3 = vmulq_n_f32(C3,dy3);
            TC2 = vaddq_f32(TC2,TC3);

            float32x4_t TC0 = vaddq_f32(C1,TC2);
            float32x4_t V = vmulq_f32(TC0,XVALUES);
            float32x2_t VW = vadd_f32(vget_low_f32(V),vget_high_f32(V));
            *dDy = vget_lane_f32(vpadd_f32(VW,VW),0)*this->yM;

        }
        // float32x4_t DXN = vmovq_n_f32(0);
        // DXN = vsetq_lane_f32(1,DXN,1);
        // DXN = vsetq_lane_f32(2*x,DXN,2);
        // DXN = vsetq_lane_f32(3*x*x,DXN,3);

        // Get DX VALUES.

        float32x4_t DXVALUES;
        {
            // Derivative of QMAT interpolation polynomials 
            // DXVALUES[0..3] = QN'[0..3](x)
            // evaluate with x^n terms replaced with their 
            // derivatives: 0, 1, 2*x, 3*x^3

            // float32x4_t X0 = vmulq_n_f32(QMAT.val[0],0);
            // float32x4_t X1 = vmulq_n_f32(QMAT.val[1],1);
            float32x4_t X2 = vmulq_n_f32(QMAT.val[2],2*x);
            float32x4_t X3 = vmulq_n_f32(QMAT.val[3],3*x2);
            X2 = vaddq_f32(X2,X3);
            DXVALUES = vaddq_f32(QMAT.val[1],X2);
        }
        
        // *dDx = lagrangeInterpolate(C0,C1,C2,C3,DXVALUES,1,y,y2,y3)*this->xM;
        {
            // interpolate dDx = lagrange(DXVALUES[0..3],Y)

            float32x4_t TC1 = vmulq_n_f32(C1,y);
            float32x4_t TC2 = vmulq_n_f32(C2,y2);
            float32x4_t TC3 = vmulq_n_f32(C3,y3);
            float32x4_t TC0 = vaddq_f32(C0,TC1);

            TC2 = vaddq_f32(TC2,TC3);
            TC0 = vaddq_f32(TC0,TC2);
            float32x4_t V = vmulq_f32(TC0,DXVALUES);
            float32x2_t VW = vadd_f32(vget_low_f32(V),vget_high_f32(V));
            *dDx = vget_lane_f32(vpadd_f32(VW,VW),0)*this->xM;

        }

    }
}



#endif
