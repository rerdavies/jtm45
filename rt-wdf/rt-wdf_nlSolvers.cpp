/*
 ==============================================================================

 This file is part of the RT-WDF library.
 Copyright (c) 2015,2016 - Maximilian Rest, Ross Dunkel, Kurt Werner.

 Permission is granted to use this software under the terms of either:
 a) the GPL v2 (or any later version)
 b) the Affero GPL v3

 Details of these licenses can be found at: www.gnu.org/licenses

 RT-WDF is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 -----------------------------------------------------------------------------
 To release a closed-source product which uses RT-WDF, commercial licenses are
 available: write to rt-wdf@e-rm.de for more information.

 ==============================================================================

 rt-wdf_nlSolvers.cpp
 Created: 2 Dec 2015 4:08:19pm
 Author:  mrest

 ==============================================================================
 */

#include "rt-wdf_nlSolvers.h"
#include <chrono>
#include <arm_neon.h>

#define ARM_NEON

#include "rt-wdf_arm_optimizations.h"


using namespace std::chrono;


//==============================================================================
// Profiling info.
//==============================================================================

using Clock = std::chrono::high_resolution_clock;
Clock::time_point startTime = Clock::now();

extern size_t g_slowInterpolations;

#undef TIMING
#ifdef TIMING

using Clock = std::chrono::high_resolution_clock;
Clock::time_point lastReport = startTime;
std::chrono::nanoseconds calculateNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds matrixNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds jMulNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds fMulNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds fnlMulNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds solverNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds invertNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds tMinNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds preambleNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds solutionNs = std::chrono::nanoseconds::zero();
std::chrono::nanoseconds report_interval = std::chrono::seconds(1);


Clock::time_point solverStartTime;
size_t totalIterations;
size_t iterationCount;

inline void onEnterSolver()
{
    solverStartTime = Clock::now();
}
inline void onExitSolver(int iter) {
    Clock::time_point tNow = Clock::now();
    auto dt = tNow -solverStartTime;
    solverNs += dt;
    totalIterations += iter+1;
    ++iterationCount;
}

void StartTiming() {
    startTime = lastReport = Clock::now();
}
void TimingReport() 
{
    Clock::time_point tNow = Clock::now();
    std::chrono::duration<rt_float> solverTimeDuration = solverNs;

    cout << "nlSolve: " << std::chrono::duration<rt_float>(solverNs).count() << endl;
    cout << "  Invert: " << std::chrono::duration<rt_float>(invertNs).count() << "  " << std::chrono::duration<rt_float>(invertNs-tMinNs).count()  <<  endl;
    cout << "  Calc: " << std::chrono::duration<rt_float>(calculateNs).count() << "  " << std::chrono::duration<rt_float>(calculateNs-tMinNs).count() << endl;
    cout << "  Matrix: " << std::chrono::duration<rt_float>(matrixNs).count() << endl;
    cout << "     fnlMul: " << std::chrono::duration<rt_float>(fnlMulNs).count() << "  " << std::chrono::duration<rt_float>(fnlMulNs-tMinNs).count()  << endl;
    cout << "     JMul: " << std::chrono::duration<rt_float>(jMulNs).count() << "  " << std::chrono::duration<rt_float>(jMulNs-tMinNs).count()  << endl;
    cout << "     FMul: " << std::chrono::duration<rt_float>(fMulNs).count() << "  " << std::chrono::duration<rt_float>(fMulNs-tMinNs).count()  << endl;
    cout << "  preamble: " << std::chrono::duration<rt_float>(preambleNs).count() << "  " << std::chrono::duration<rt_float>(preambleNs-tMinNs/6).count()  << endl;
    cout << "  soln: " << std::chrono::duration<rt_float>(solutionNs).count() << "  " << std::chrono::duration<rt_float>(solutionNs-tMinNs/6).count()  << endl;

    std::chrono::duration<rt_float> tTotal = tNow-startTime;
    std::chrono::duration<rt_float> tAccountedFor = 
    (
        (invertNs-tMinNs) + (calculateNs-tMinNs) + (fnlMulNs-tMinNs) + 
        (jMulNs-tMinNs) + (fMulNs-tMinNs) + (solutionNs-tMinNs/6) +
        (preambleNs-tMinNs/6)
    );
    lastReport = tNow;
    cout << "Total: " << tTotal.count() <<  " accounted for: " << tAccountedFor.count() << endl;
    cout << "TMin: " << std::chrono::duration<rt_float>(tMinNs).count() << endl;
    cout << "Avg Iterations: " << ((rt_float)totalIterations)/iterationCount << " n: " << iterationCount << endl;
    cout << "Slow interpolations: " << g_slowInterpolations << endl;

    cout << endl;
}
#else
inline void onEnterSolver()
{

}
inline void onExitSolver(int iter) {

}

void StartTiming() {
    startTime = Clock::now();
}

void TimingReport() 
{
    std::chrono::duration<rt_float> tTotal = Clock::now()-startTime;
    cout << "Total: " << tTotal.count() << endl;
    cout << "Slow interpolations: " << g_slowInterpolations << endl;
}

#endif

//==============================================================================
// Parent class for nlSolvers
//==============================================================================
nlSolver::nlSolver( ) : numNLPorts( 0 ) {

}

nlSolver::~nlSolver( ) {

}

//----------------------------------------------------------------------
int nlSolver::getNumPorts( ) {
    return numNLPorts;
}

// public for debug and test purposes.
bool g_UseInterpolatedTriode = true;

//==============================================================================
// Newton Solver
//==============================================================================
nlNewtonSolver::nlNewtonSolver( std::vector<int> nlList,
                        matData* myMatData ) : myMatData ( myMatData ) {

    // set up Vec<nlModel> nlModels properly according to std::vector<int> nlList
    for( int nlModel : nlList )
    {
        switch( nlModel ) {
            // Diodes:
            case DIODE:             // single diode
            {
                nlModels.push_back(new diodeModel);
                break;
            }
            case DIODE_AP:          // antiparallel diode pair
            {
                nlModels.push_back(new diodeApModel);
                break;
            }
            // Bipolar Transistors:
            case NPN_EM:            // Ebers-Moll npn BJT
            {
                nlModels.push_back(new npnEmModel);
                break;
            }
            // Triode Tubes:
            case TRI_DW:            // Dempwolf triode model
            {
                if (g_UseInterpolatedTriode)
                {
                    nlModels.push_back(new interpolatedTriDwModel);
                } else {
                    nlModels.push_back(new triDwModel);
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }

    numNLPorts = 0;
    for ( nlModel* model : nlModels ) {
        numNLPorts += model->getNumPorts();
    }

    x0       = new Col<rt_float>(numNLPorts, fill::zeros);
    F        = new Col<rt_float>(numNLPorts, fill::zeros);
    J        = new Mat<rt_float>(numNLPorts,numNLPorts, fill::zeros);
    fNL      = new Col<rt_float>(numNLPorts, fill::zeros);
    JNL      = new Mat<rt_float>(numNLPorts,numNLPorts, fill::zeros);
    Fmat_fNL = new Col<rt_float>(numNLPorts, fill::zeros);
    p = new Col<rt_float>(numNLPorts,fill::zeros);
    xnew = new Col<rt_float>(numNLPorts,fill::zeros);
    #ifdef ARM_NEON
    JInv     = new Mat<rt_float>(numNLPorts,numNLPorts, fill::zeros);
    FTemp        = new Col<rt_float>(numNLPorts, fill::zeros);
    STemp = new Col<rt_float>(17,fill::zeros);
    #endif

}

nlNewtonSolver::~nlNewtonSolver( ) {
    size_t modelCount = nlModels.size();
    for( size_t i = 0; i < modelCount; i++ ) {
        delete nlModels[i];
    }
    delete x0;
    delete F;
    delete p;
    delete J;
    delete fNL;
    delete JNL;
    delete Fmat_fNL;
}


//----------------------------------------------------------------------
void nlNewtonSolver::nlSolve( Col<rt_float>* inWaves,
                          Col<rt_float>* outWaves ) {

    onEnterSolver();

    #ifdef TIMING
        Clock::time_point tPre0= Clock::now();
    #endif


    rt_float iter = 0;            // # of iteration
    rt_float alpha = 0;

    (*J).zeros();

    if ( firstRun ) {
        firstRun = false;
    }
    else {
        #ifdef ARM_NEON
            neon_matrix_x_vec(x0,&myMatData->Emat,inWaves,8,17);
            neon_add_vec8(x0,x0,Fmat_fNL);
        #else
                    // 8x1      8x17  1x17
        (*x0) = (*Fmat_fNL) + (myMatData->Emat)*(*inWaves);
        #endif
    }

    evalNlModels( inWaves, myMatData, x0 );


    rt_float normF = norm(*F);
    //printf("iter alpha         ||F||_2\n");
    //printf(" %3g %9.2e %14.7e\n", iter, alpha, normF);

    // Col<rt_float> xnew;
    rt_float normFnew;
    #ifdef TIMING
        preambleNs += Clock::now()-tPre0;
    #endif

    while ( (normF >= TOL) && (iter < ITMAX) )
    {
        #ifdef TIMING
            Clock::time_point t0= Clock::now();
        #endif
        #ifdef JUNK
            cout << "J" << endl;
            (*J).print();
            cout << endl;
            (*J).i().print();
            cout << endl;
            (*F).print();
            cout << endl;
        #endif

        #ifdef ARM_NEON

            neon_invert_88(J,JInv); // destructive,


            *p = (*JInv) * (*F);

            //xnew = (*x0) - p;
            neon_sub_vec8(xnew,x0,p);
            
        #else 
            (*p) = - (*J).i() * (*F);
            alpha = 1;
            xnew = (*x0) + alpha * (*p);
        #endif

        #ifdef TIMING
            invertNs += Clock::now() - t0;
        #endif




        evalNlModels(inWaves, myMatData, xnew);
        normFnew = norm(*F);
        (*x0) = *xnew;
        normF = normFnew;
        // cout << "normF: " << normF << endl;
        iter++;

    //        printf(" %3g %9.2e %14.7e\n", iter, alpha, normF);
    }

    #ifdef TIMING
        Clock::time_point tSol0= Clock::now();
    #endif

#ifdef ARM_NEON
    neon_matrix_x_vec(outWaves,&(myMatData->Mmat),inWaves);
    neon_matrix_x_vec(STemp,&myMatData->Nmat,fNL);
    neon_add_vec_17(outWaves,outWaves,STemp);
#else
    (*outWaves) = (myMatData->Mmat) * (*inWaves) + (myMatData->Nmat) * (*fNL);
#endif

    #ifdef TIMING
        solutionNs += Clock::now()-tSol0;
    #endif

    onExitSolver(iter);

}

//----------------------------------------------------------------------
void nlNewtonSolver::evalNlModels( Col<rt_float>* inWaves,
                               matData* myMatData,
                               Col<rt_float>* x ) {
    int currentPort = 0;
    #ifdef ARM_NEON
        neon_zero(JNL);
    #else 
        (*JNL).zeros();
    #endif

#ifdef TIMING
    Clock::time_point t0= Clock::now();
#endif

    for ( nlModel* model : nlModels ) {
        model->calculate( fNL, JNL, x, &currentPort );
    }
#ifdef TIMING
    Clock::time_point t1= Clock::now();
#endif

    #ifdef ARM_NEON
        neon_multiply_881(Fmat_fNL,&(myMatData->Fmat),fNL);
    #else
        (*Fmat_fNL) = myMatData->Fmat*(*fNL);
    #endif
    #ifdef TIMING
        Clock::time_point tfnl= Clock::now();
        fnlMulNs += tfnl-t1;
    #endif


    // o 8x17  + 8 + 8
    #ifdef ARM_NEON
        neon_matrix_x_vec(FTemp,&myMatData->Emat,inWaves,8,17);
        //(*F) = (*FTemp) + (*Fmat_fNL) - (*x);
        neon_add_vec8(F,FTemp,Fmat_fNL);
        neon_sub_vec8(F,F,x);
    #else 
        (*F) = (myMatData->Emat)*(*inWaves) + (*Fmat_fNL) - (*x);
    #endif

    #ifdef TIMING
        Clock::time_point tf= Clock::now();
        fMulNs += tf-tfnl;
    #endif


    #ifdef JMUL_OPT
        // o 8x8x8 + 8x8   (could be 8x8x2+8) Banded Matrix!!!
        (*J) = (myMatData->Fmat)*(*JNL) - eye(size(*JNL)); // profile: 1.28571
    #else
        // o 4x4x6 +4
        int rows, columns;
        rows = columns = (*JNL).n_rows;
        for (int r = 0; r < rows; r += 2)
        {
            for (int c = 0; c < columns; c += 2)
            {
                rt_float c00 = (*JNL)(c,c);
                rt_float c01 = (*JNL)(c,c+1);
                rt_float c11 = (*JNL)(c+1,c+1);
            
                (*J)(r,c) = myMatData->Fmat(r,c)*c00;
                (*J)(r,c+1) = myMatData->Fmat(r,c)*c01 + myMatData->Fmat(r,c+1)*c11;
                (*J)(r+1,c) = myMatData->Fmat(r+1,c)*c00;
                (*J)(r+1,c+1) = myMatData->Fmat(r+1,c)*c01 + myMatData->Fmat(r+1,c+1)*c11;
            }
        }
        for (int r = 0; r < rows; ++r)
        {
            (*J)(r,r) -= 1;
        }
    #endif

    #ifdef TIMING
        Clock::time_point tj = Clock::now();
        jMulNs += tj-tf;
    #endif

#ifdef TIMING
    Clock::time_point t2= Clock::now();
    tMinNs += t2-tj;

    calculateNs += t1-t0;

    matrixNs += t2-t1;

#endif

}

