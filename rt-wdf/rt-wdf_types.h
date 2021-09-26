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

    rt-wdf_types.h
    Created: 15 Dec 2015 3:54:27pm
    Author:  mrest

  ==============================================================================
*/

#ifndef RTWDF_TYPES_H_INCLUDED
#define RTWDF_TYPES_H_INCLUDED
#include <stddef.h>
#include <stdlib.h>

constexpr size_t MEMORY_ALIGNMENT = 16;

#define ALIGNED_16 __attribute__((aligned (MEMORY_ALIGNMENT)))
#define ALIGNED_8  __attribute__((aligned (8)))

inline void*alignedAlloc(size_t size)
{
    size_t roundedSize = (size+MEMORY_ALIGNMENT-1)/MEMORY_ALIGNMENT*MEMORY_ALIGNMENT;
    return aligned_alloc(16,roundedSize);
}

template <typename TYPE>
TYPE*alignedAlloc(size_t size) {
    return (TYPE*)alignedAlloc(sizeof(TYPE)*size);
}

#include <armadillo>
using namespace arma;

typedef float rt_float;



//==============================================================================
/** A struct that holds matrices for R-type and NL root nodes.

    wdfRootRtype only uses the S matrix of this struct.
    wdfRootNL only uses the E,F,M,N matrices of this struct.

    @see wdfRootRtype, wdfRootNL
*/
typedef struct matData{

    /** S-Matrix as used in wdfRootRtype.
        Size: (numBrPorts) x (numBrPorts)
    */
    Mat<rt_float> Smat;

    /** E-Matrix as used in wdfRootNL.
        Size: (numNlPorts) x (numBrPorts)
    */
    Mat<rt_float> Emat;

    /** F-Matrix as used in wdfRootNL.
        Size: (numNlPorts) x (numNlPorts)
    */
    Mat<rt_float> Fmat;

    /** M-Matrix as used in wdfRootNL.
        Size: (numBrPorts) x (numBrPorts)
    */
    Mat<rt_float> Mmat;

    /** N-Matrix as used in wdfRootNL.
        Size: (numBrPorts) x (numNlPorts)
    */
    Mat<rt_float> Nmat;

    /** T-Matrix as used in wdfRootLinear and wdfRootMixed.
     Size: (numBrPorts) x (numNlPorts)
     */
    Mat<rt_float> Tmat;

} matData;

typedef enum paramType {
    boolParam,
    doubleParam
} paramType;

typedef struct paramData {
    std::string name;
    size_t ID;
    paramType type;
    rt_float value;
    std::string units;
    rt_float lowLim;
    rt_float highLim;
} paramData;

#endif  // RTWDF_TYPES_H_INCLUDED
