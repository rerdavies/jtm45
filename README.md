# jtm45 Amp Emulation LV2 Plugin

A JTM45 Amp Emulator LV2 Plugin for Raspberry PI 4 devices running a 64-bit operting system. The emulator uses real-time WDF analog modelling techniques to model JTM45 power stage circuitry using code developed and published by  Maximilian Rest, W. Ross Dunkel, Kurt James Werner and Julius O. Smith. 

An exceptionally beautiful tube amp power stage emulation based on the work of Maximilian Rest, W. Ross Dunkel, Kurt James Werner and Julius O. Smith. This particular project consists of a highly-optimized port to ARM aarch64 architecture, which converts time-critical parts of the algorithm to ARM neon.

Note that the JTM45 power stage differs only in very small details from a Fender Bassman.


Typically,  you will want to place a tone-stack plugin before the jtm45 plugin, and a cabinet simulator plugin after the jtm45 plugin. See the [ToobAmp](https:://github.com/rerdavies/ToobAmp) project for accurate light-weight emulations of Fender and Marshal tone stacks and cab sims.

Based on a preliminary Lv2 plugin written by Massimo Pennazio which can be found  [here](https://github.com/MaxPayne86/rt-wdf_lv2)

That project is in turn based on RT-WDF source code presented in a [paper](http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf) by Maximilian Rest, W. Ross Dunkel, Kurt James Werner and Julius O. Smith.^1 See the References section for a full citation.

## Performance Requirements

This LV2 plugin is highly CPU intensive. To run in realtime, you must meet the following requirements:

* ARM Cortex A72 1.5GHz or better, running 64-bit mode (aarch64)
* A 64-bit Linux operating system, with a PREEMPT or RT_PREEMPT kernel.

jtm45 can be compiled on ARM aarch32, but it runs about 20% slower, so you will need an exceptionally good processor to run in realtime on a 32-bit operating system. On a Raspberry Pi 4B running 32-bit Raspbian, overclocking is not sufficient to run in real time

jtm45 was built and tested on the following operating system:

* Ubutunu 21.04 Gnome3 desktop (KDE wil NOT work)
* Ubutunto Studio additions instlled using APT
* All Ubuntu studio performance options applied.
* PulseAudio removed (sudo apt remove pulseaudio)'
* Hosted with the  [Pipedal](https://github.com/rerdavies/pipedal) Web/IOT interface for Raspberry Pi.

jtm45 was built and tested on the following hardware:

* Raspberry Pi 4/B with 8GB of RAM. (2GB is probably sufficient to run, 4GB is probably required to build)
* External MOTU M2 USB Audio Adapter, running at 48khz 32 samples/3 frames (4ms latency)


 
## Building jtm45

__Important note: the rt-wdf files included in this project were modified while optimizing jtm45 code for ARM neon, and are NO LONGER SUITABLE FOR GENERAL PURPOSE USE. Please look elsewhere for rt-wdf source files I believe the most current branch of rt-wdf source files are found in the JUICE audio library._

The following packages must be pre-instaled:
     
     sudo apt install libarmadillo-dev
     sudo apt install lv2-dev

If you are not using Visual Sstudio Code, you will need to install a reasonably current build of CMake.

jtm45 was developed using Visual Studio Code, with CMake plugins installed. To build the project with Visual Studio Code, load the project folder in Visual Studio Code. Code will autodetect the project's CMake build system. Once loaded, you can configure and build the project using integrated CMake build tools on the bottom toolbar of the Visual Studio Code window.

After building, run the following shell script in the project root to install the jtm45 plugin.

    `sudo ./install`
    
If you are not using Visual Studio Code, the following shell scripts can be used to configure and build the project:

	./config
	./bld
	
## References

1. Maximilian Rest, W. Ross Dunkel, Kurt James Werner, Julius O. Smith, “RT-WDF—A MODULAR WAVE DIGITAL FILTER LIBRARY WITH SUPPORT FOR
ARBITRARY TOPOLOGIES AND MULTIPLE NONLINEARITIES,” in Proceedings of the 19th International Conference on Digital Audio Effects (DAFx-16), Brno, Czech Republic, September 5–9, 2016. [http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf](http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf)

