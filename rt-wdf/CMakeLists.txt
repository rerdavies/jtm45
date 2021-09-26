cmake_minimum_required(VERSION 3.16.0)
set(VERBOSE 1)

include(FindPkgConfig)

set(CMAKE_INSTALL_SO_EXE false)

find_package(Armadillo REQUIRED)

# specify the C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX /usr)


# set (ARM_NEON_CFLAGS "-mfpu=neon -funsafe-math-optimizations  -mfloat-abi=hard")
set (ARM_NEON_CFLAGS "-mcpu=cortex-a72 -mtune=cortex-a72 -funsafe-math-optimizations  -mfloat-abi=hard -mfpu=neon-fp-armv8 -ftree-vectorize -fPIC ")
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-psabi -Werror ${ARM_NEON_CFLAGS}"  )

include_directories((${ARMADILLO_INCLUDE_DIRS}))
include_directories((${ARMADILLO_INCLUDE_DIRS}))
#################################
add_library(rt-wdf STATIC
        matrix.h
        matrix_arm_neon.h
        rt-wdf_arm_optimizations.h
        rt-wdf.cpp
        rt-wdf_nlModels.cpp
        rt-wdf_nlSolvers.cpp
        rt-wdf.h
        rt-wdf_nlModels.h
        rt-wdf_nlSolvers.h
        rt-wdf_types.h
        rt-wdf_interpolatedTable.h
        
    )

add_compile_definitions(ARMA_DONT_USE_WRAPPER ARMA_NO_DEBUG )
target_link_libraries(rt-wdf PUBLIC lapack blas)

#INSTALL(TARGETS rt-wdf
#    LIBRARY DESTINATION lib
#    )

#INSTALL(FILES 
#    rt-wdf.h
#    rt-wdf_nlModels.h
#    rt-wdf_nlSolvers.h
#    rt-wdf_types.h
#    rt-wdf_arm_optimizations.h
#    DESTINATION include/
#)







