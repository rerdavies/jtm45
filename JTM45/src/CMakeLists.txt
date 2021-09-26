cmake_minimum_required(VERSION 3.19)

include(FindPkgConfig)
# set (ARM_NEON_CFLAGS "-mfpu=neon -funsafe-math-optimizations  -mfloat-abi=hard")
set (ARM_NEON_CFLAGS "-mcpu=cortex-a72 -mtune=cortex-a72 -funsafe-math-optimizations  -mfloat-abi=hard -mfpu=neon-fp-armv8 -ftree-vectorize ")
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-psabi -Werror ${ARM_NEON_CFLAGS}"  )
set (CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} )

add_executable(jtm45test 
    testMain.cpp
    testPerformance.cpp
    testInterpolationTables.cpp
    testInvert.cpp
    testNlModels.cpp
    testMatrixMultiply.cpp
    testMatrix.cpp
    AudioFile.h
)


target_include_directories(jtm45test PRIVATE
    "${JTM45_SOURCE_DIR}/rt-wdf"
)


add_compile_definitions(jtm45test ARMA_DONT_USE_WRAPPER)

target_link_libraries(jtm45test 
    rt-wdf blas lapack
    profiler

    )

add_library(jtm45 SHARED
    JTM45.cpp
    wdfJTM45Tree.hpp
)

target_include_directories(jtm45 PRIVATE
    "${JTM45_SOURCE_DIR}/rt-wdf"
)


add_compile_definitions(jtm45 ARMA_DONT_USE_WRAPPER)

target_link_libraries(jtm45 
    rt-wdf blas lapack
    )

set(INSTALL_PATH /usr/local/lib/lv2)
set(COMPLETE_INSTALL_PATH "${INSTALL_PATH}/jtm45.lv2")


set_target_properties(jtm45 PROPERTIES OUTPUT_NAME "jtm45")

set_target_properties(jtm45 PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(jtm45 PROPERTIES SOVERSION 0)

set_target_properties(jtm45 PROPERTIES OUTPUT_NAME "jtm45")
set_target_properties(jtm45 PROPERTIES PREFIX "")

install(TARGETS jtm45
    LIBRARY DESTINATION "${COMPLETE_INSTALL_PATH}"
    )

# Copy all assets to resources file
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../ttl/ DESTINATION "${COMPLETE_INSTALL_PATH}"
        )    


