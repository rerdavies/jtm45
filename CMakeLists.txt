cmake_minimum_required(VERSION 3.19)

project(JTM45 VERSION 0.0.1 DESCRIPTION "JTM45 Amp Emulation LV2 Plugin")


add_subdirectory(rt-wdf)

add_subdirectory(
    JTM45/src
)