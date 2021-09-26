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

 rt-wdf_nlModels.cpp
 Created: 2 Dec 2015 4:10:47pm
 Author:  mrest

 ==============================================================================
*/

#include "rt-wdf_nlModels.h"

#include "rt-wdf_arm_optimizations.h"

//==============================================================================
// Parent class for nlModels
//==============================================================================
nlModel::nlModel( int numPorts ) : numPorts (numPorts) {

}

nlModel::~nlModel( ) {

}

//----------------------------------------------------------------------
int nlModel::getNumPorts( ) {
    return numPorts;
}


//==============================================================================
// Diode Models according to Kurt Werner et al
// ("An Improved and Generalized Diode Clipper Model for Wave Digital Filters")
//==============================================================================
#define Is_DIODE    2.52e-9
#define VT_DIODE    0.02585

diodeModel::diodeModel() : nlModel( 1 ) {

}

//----------------------------------------------------------------------
void diodeModel::calculate( Col<rt_float>* fNL,
                            Mat<rt_float>* JNL,
                            Col<rt_float>* x,
                            int* currentPort ) {

    const rt_float vd = (*x)(*currentPort);
    const rt_float arg1 = vd/VT_DIODE;

    (*fNL)(*currentPort) = Is_DIODE*(exp(arg1)-1);
    (*JNL)(*currentPort,*currentPort) = (Is_DIODE/VT_DIODE)*exp(arg1);

    (*currentPort) = (*currentPort)+getNumPorts();
}

//==============================================================================
diodeApModel::diodeApModel( ) : nlModel( 1 ) {

}

//----------------------------------------------------------------------
void diodeApModel::calculate( Col<rt_float>* fNL,
                              Mat<rt_float>* JNL,
                              Col<rt_float>* x,
                              int* currentPort) {

    const rt_float vd = (*x)(*currentPort);
    const rt_float arg1 = vd/VT_DIODE;

    (*fNL)(*currentPort) = Is_DIODE*(exp(arg1)-1)-Is_DIODE*(exp(-arg1)-1);
    (*JNL)(*currentPort,*currentPort) = (Is_DIODE/VT_DIODE)*(exp(arg1)+exp(-arg1));

    (*currentPort) = (*currentPort)+getNumPorts();
}


//==============================================================================
// Transistor Models using Ebers-Moll equations
// ("Large-signal behavior of junction transistors")
//==============================================================================
#define Is_BJT      5.911e-15
#define VT_BJT      0.02585
#define BETAF       1.434e3
#define BETAR       1.262
#define ALPHAF      (BETAF/(1.0+BETAF))     //TAKE CARE OF ( ) TO COMPILE CORRECTLY!!!!!! ARGHH!!
#define ALPHAR      (BETAR/(1.0+BETAR))     //TAKE CARE OF ( ) TO COMPILE CORRECTLY!!!!!!


npnEmModel::npnEmModel() : nlModel( 2 ) {

}

//----------------------------------------------------------------------
void npnEmModel::calculate( Col<rt_float>* fNL,
                            Mat<rt_float>* JNL,
                            Col<rt_float>* x,
                            int* currentPort) {

    const rt_float vBC = (*x)(*currentPort);
    const rt_float vBE = (*x)((*currentPort)+1);

    const rt_float vBC_o_VT_BJT = vBC/VT_BJT;
    const rt_float vBE_o_VT_BJT = vBE/VT_BJT;
    const rt_float Is_BJT_o_VT_BJT = Is_BJT/VT_BJT;
    const rt_float Is_BJT_o_ALPHAR = Is_BJT/ALPHAR;
    const rt_float Is_BJT_o_ALPHAF = Is_BJT/ALPHAF;


    (*fNL)(*currentPort) = -Is_BJT*(exp(vBE_o_VT_BJT )-1)+(Is_BJT_o_ALPHAR)*(exp(vBC_o_VT_BJT)-1);
    (*JNL)((*currentPort),(*currentPort)) = (Is_BJT_o_ALPHAR/VT_BJT)*exp(vBC_o_VT_BJT);
    (*JNL)((*currentPort),((*currentPort)+1)) = (-Is_BJT_o_VT_BJT)*exp(vBE_o_VT_BJT );

    (*fNL)((*currentPort)+1) = (Is_BJT_o_ALPHAF)*(exp(vBE_o_VT_BJT )-1)-Is_BJT*(exp(vBC_o_VT_BJT)-1);
    (*JNL)(((*currentPort)+1),(*currentPort)) = (-Is_BJT_o_VT_BJT)*exp(vBC_o_VT_BJT);
    (*JNL)(((*currentPort)+1),((*currentPort)+1)) = (Is_BJT_o_ALPHAF/VT_BJT)*exp(vBE_o_VT_BJT );

    (*currentPort) = (*currentPort)+getNumPorts();
}


//==============================================================================
// Triode model according to Dempwolf et al
// ("A physically-motivated triode model for circuit simulations")
//==============================================================================
triDwModel::triDwModel() : nlModel( 2 ) {


}

//----------------------------------------------------------------------
void triDwModel::calculate( Col<rt_float>* fNL,
                            Mat<rt_float>* JNL,
                            Col<rt_float>* x,
                            int* currentPort) {

    //  Profile: ~ 200

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

    const double vAC = (*x)(*currentPort);
    const double vGC = (*x)((*currentPort)+1);
    const double vAC_mu =  vAC* InvMu;


    const double exp_Cg_vGC = exp( Cg * vGC );
    const double log_1_exp_Cg_vGC_Cg = (log( 1 + exp_Cg_vGC ) * InvCg);

    double pow_log_1_exp_Cg_vGC_Cg_E_1 = pow( log_1_exp_Cg_vGC_Cg, (E-1));
    // Ig
    //* (*fNL)((*currentPort)+1) = Gg * pow( log_1_exp_Cg_vGC_Cg, E ) + Ig0;
    (*fNL)((*currentPort)+1) = Gg * pow_log_1_exp_Cg_vGC_Cg_E_1*log_1_exp_Cg_vGC_Cg + Ig0;    
    // dIg / dvAC
    (*JNL)(((*currentPort)+1),(*currentPort)) = 0;
    // dIg / dvGC
        //* (*JNL)(((*currentPort)+1),((*currentPort)+1)) = ( (Gg * E) * exp_Cg_vGC *
        //*                                                  pow( log_1_exp_Cg_vGC_Cg, (E-1)) ) /
        //*                                                (1 + exp_Cg_vGC);

    (*JNL)(((*currentPort)+1),((*currentPort)+1)) = ( (Gg * E) * exp_Cg_vGC *
                                                        pow_log_1_exp_Cg_vGC_Cg_E_1 ) /
                                                        (1 + exp_Cg_vGC);


    const double exp_C_vAC_mu_vGC = exp( C * ( vAC_mu + vGC ));
    double log_1_exp_C_vAC_mu_vGC_InvC = (log(1 + exp_C_vAC_mu_vGC) * InvC);
    double pow_log_1_exp_C_vAC_mu_vGC_C_y_1 = pow( log_1_exp_C_vAC_mu_vGC_InvC, (Y-1));

    // Ik 
        //* (*fNL)(*currentPort) = G * pow( log( 1 + exp_C_vAC_mu_vGC ) *  InvC , Y ) - (*fNL)((*currentPort)+1);
    (*fNL)(*currentPort) = G * pow_log_1_exp_C_vAC_mu_vGC_C_y_1*log_1_exp_C_vAC_mu_vGC_InvC - (*fNL)((*currentPort)+1);

    // dIk / dvAC
    (*JNL)((*currentPort),(*currentPort)) = ( G * Y * exp_C_vAC_mu_vGC *
                                              pow_log_1_exp_C_vAC_mu_vGC_C_y_1 ) /
                                            (Mu * (1 + exp_C_vAC_mu_vGC));
    // dIk / dvGC
    (*JNL)((*currentPort),((*currentPort)+1)) = ( G * Y * exp_C_vAC_mu_vGC *
                                                  pow_log_1_exp_C_vAC_mu_vGC_C_y_1 ) /
                                                (1 + exp_C_vAC_mu_vGC) - (*JNL)(((*currentPort)+1),((*currentPort)+1));

#ifdef JUNK
    std::cout <<"vAC: " << vAC << " vGC: " << vGC << endl;

    std::cout << "Ik: " << (*fNL)(*currentPort) 
        << " dIk/dvAC: " << (*JNL)((*currentPort),(*currentPort)) 
        << " dIk/dvGC: " << (*JNL)((*currentPort),((*currentPort)+1))
        << " Ig: " << (*fNL)((*currentPort)+1) 
        << " dIg/dvGC: " << (*JNL)(((*currentPort)+1),(*currentPort)) 
        << endl;
#endif

    (*currentPort) = (*currentPort)+getNumPorts();

}



//==============================================================================
// Triode model using lagrange interpolated tables to improve realtime performance.
// Values in the table are from the Dempwolf triode model.
// ("A physically-motivated triode model for circuit simulations")
//==============================================================================

// Prototype function for use by Interpolator.
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
// Prototype function for use by Interpolator.
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


interpolatedTriDwModel::interpolatedTriDwModel() 
: 
    vGCtoIg(2048,MIN_VGC*2050/2048,MAX_VGC*2050/2048,TriodeModelIg),
    vACvGCtoIk(
        MIN_VAC,MAX_VAC,
        MIN_VGC,MAX_VGC, TriodeModelIk)
{

}

//----------------------------------------------------------------------

size_t g_slowInterpolations = 0;
void interpolatedTriDwModel::calculate( Col<rt_float>* fNL,
                            Mat<rt_float>* JNL,
                            Col<rt_float>* x,
                            int* currentPort) {
    
    //  Profile: ~ 200

    rt_float vAC = (*x)(*currentPort);
    rt_float vGC = (*x)((*currentPort)+1);


    // Wild voltage swings during init. Just delegate them to the original code.
    if (vAC <= MIN_VAC || vAC >= MAX_VAC
    || vGC <= MIN_VGC || vGC >= MAX_VGC)
    {
        ++g_slowInterpolations;
        //cout << "slow: " << vAC << "," << vGC << endl;
        triDwModel::calculate(fNL,JNL,x,currentPort);
        return;
    }


    // cout <<"vAC: " << vAC << " vGC: " << vGC << endl;

    rt_float Ik, dIk_dvAC, dIk_dvGC;

    vACvGCtoIk.get(vAC,vGC,&Ik,&dIk_dvAC,&dIk_dvGC);

    rt_float Ig,dIg_dVGC;
    vGCtoIg.get(vGC,&Ig,&dIg_dVGC);
    

        
    // Ig
    (*fNL)((*currentPort)+1) = Ig;
    // dIg / dvAC
    (*JNL)(((*currentPort)+1),(*currentPort)) = 0;
    // dIg / dvGC
    (*JNL)(((*currentPort)+1),((*currentPort)+1)) = dIg_dVGC;

    // Ik 
    (*fNL)(*currentPort) = Ik;

    // dIk / dvAC
    (*JNL)((*currentPort),(*currentPort)) = dIk_dvAC;
    // dIk / dvGC
    (*JNL)((*currentPort),((*currentPort)+1)) = dIk_dvGC;


    // cout << "Ik: " << Ik << " dIk/dvAC: " << dIk_dvAC << " dIk/dvGC: " << dIk_dvGC 
    //     << " Ig: " << Ig << " dIg/dvGC: " << dIg_dVGC << endl;

    (*currentPort) = (*currentPort)+getNumPorts();

}

