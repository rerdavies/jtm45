# jtm45 Amp Emulation LV2 Plugin (ARM aarch64)

*It has come to my attention that the base files used in this project use data that has not been generated for the correct sampling rate (it's generated for 4x oversampling at 48KHz). Unfortunately, the code used to generate the data has not been open-sourced. I'm working on a (fairly substantial) solution to this. Please stay tuned.*

*In the meantime, this plugin makes a very pretty noise, but not one that a jtm45 would make. Modify your expectations accordingly.*

A JTM45 Amp Emulator LV2 Plugin for Raspberry PI 4 devices running a 64-bit operating system. The emulator uses real-time WDF analog modelling techniques to model JTM45 power stage circuitry using code developed and published by  Maximilian Rest, W. Ross Dunkel, Kurt James Werner and Julius O. Smith. 

Note that the JTM45 power stage differs only in very small details from a Fender Bassman.

The result is an exceptionally beautiful tube amp power stage emulation. This particular project consists of optimizing the code so that it runs in real time on a Raspberry Pi 4. From the original code: pretty much all of the matrix operations have been replaced with highly-optimized ARM Neon SIMD matrix operations; closed form equations for non-linear behaviour were replaced with lagrange-interpolators (again, coded for ARM Neon SIMD); conversion of all real-time math to 32-bit from 64-bit float; and minor tweaking of the Newton-Raphson solver for the non-linear parts of the algorithm; completing the LV2 integeration that had been 99% completed by Massimo Pennazio.

Typically,  you will want to place a tone-stack plugin before the jtm45 plugin, and a cabinet simulator plugin after the jtm45 plugin. See the [ToobAmp](https:://github.com/rerdavies/ToobAmp) project for accurate light-weight but linear emulations of Fender and Marshal tone stacks and cab sims.

Based on a preliminary Lv2 plugin written by Massimo Pennazio which can be found  [here](https://github.com/MaxPayne86/rt-wdf_lv2)

That project is in turn based on RT-WDF source code presented in a [paper](http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf) by Maximilian Rest, W. Ross Dunkel, Kurt James Werner and Julius O. Smith.^1 See the References section for a full citation.

## Performance Requirements

This LV2 plugin is highly CPU intensive. To run in realtime, you must meet the following requirements:

* ARM Cortex A72 1.5GHz or better, running in 64-bit mode (aarch64) (e.g., a stock Raspberry Pi 4B).
* A 64-bit Linux operating system, with a PREEMPT or RT_PREEMPT kernel.

jtm45 can be compiled on ARM aarch32, but it runs about 20% slower, so you will need an exceptionally good processor to run in realtime on a 32-bit operating system. On a Raspberry Pi 4B running 32-bit Raspbian, overclocking is not sufficient to run in real time. It's possible that the performance hit is due to spilling of neon registers; but the current GCC toochain does not provide wonderful support for code optimization. It's possible that someone with access to (incredibly expensive) ARM profiling tools could push the aarch32 binary into real-time range. (It's currently close, but not there yet). 

jtm45 was built and tested on the following operating system:

* Ubutunu 21.04 Gnome3 desktop
* Ubutunto Studio additions installed onto stock Ubuntu using APT
* All Ubuntu Studio performance options applied (applies real-time configuration tweaks, but does not provide an actual RT_PREEMPT kernel).
* PulseAudio removed (sudo apt remove pulseaudio), since there seems to be no easy way to get Jackd to run as a service when PulseAudio is installed.
* Hosted with the  [Pipedal](https://github.com/rerdavies/pipedal) Web/IOT interface for Raspberry Pi.

jtm45 was built and tested on the following hardware:

* Raspberry Pi 4/B with 8GB of RAM. (2GB is probably sufficient to run, 4GB is probably required to build)
* External MOTU M2 USB Audio Adapter, running at 48khz 32 samples/3 frames (4ms latency)

### Notes on Not Running jtm45 on aarch32

For those who are interested in pursuing further aarch32 optimization.. I was unable to get the aarch32 build of jtm45 to operate in realtime. But it is very close to being able to do so. In the end, it all runs well on Ubuntu 64, and I don't see any easy path forward on further aarch32 optimizations, so I left it at that.

jtm45 can be compiled on ARM aarch32, but it runs about 30% slower than on aarch64.  When running on Rasbian aarch32, the compiled code consumes 60% of available CPU, according to Jack. On Ubuntu 64-bit, it consumes about 40%. I think you need to get it to about 55% to run at all, and 50% to be usable (i.e. to add a couple of addition plugins). 

A faster CPU, or an overclocked CPU gains some performance but not enough. (I tried overclocking my Raspberry Pi 4 to 1.8Ghz without success). 

I am a professional developer with a huge amount of experience optimizing performance critical code; so don't expect low-hanging fruit. That being said, I don't have access to the insanely expensive ARM profiling and optimization tools that I would need to push aarm32 code to the limiit. It's possible that better compilers or better profiling tools could push the aarch32 compiled code into real-time range.

Probably the first thing you should try is to use a GCC 10.x toolchain instead of the GCC 8.3 toolchain that is default on Raspbian. GCC 8.3 instruction scheduling for ARM Cortex A72 looks credible; but I don't have tools to confirm that. The simpler test is just to compile it with GCC 10.x to see if there is a dramatic improvement. GCC 10.x is not available on current Rasbian as a package. You would either have to build the GCC 10.x toolchain, or crosscompile from a platform that has it.

There is also one more significant code optimization that I know about that could gain about 3% in overall performance. Drop me a note if you get close enough that 3% would matter.

I was unable to get a working profiler running 32-bit Rasbian (but was able to do so on 64-bit Ubuntu). Profiling the 32-bit code with a sampling profiler would be the next step. Existing optimization on aarch32 was done by benchmark instrumentation of the code, which is neccesarily inexact.

 
## Building jtm45

__Important note: the rt-wdf files included in this project were modified while optimizing jtm45 code for ARM neon, and are NO LONGER SUITABLE FOR GENERAL PURPOSE USE. Please look elsewhere for rt-wdf source files I believe the most current branch of rt-wdf source files are found in the JUICE audio library._

The following packages must be pre-instaled:
     
     sudo apt install libarmadillo-dev
     sudo apt install lv2-dev

If you are not using Visual Studio Code, you will need to install a reasonably current build of CMake.

jtm45 was developed using Visual Studio Code, with CMake plugins installed. To build the project with Visual Studio Code, load the project folder in Visual Studio Code. Code will autodetect the project's CMake build system. Once loaded, you can configure and build the project using integrated CMake build tools on the bottom toolbar of the Visual Studio Code window.

After building, run the following shell script in the project root to install the jtm45 plugin.

    `sudo ./install`
    
If you are not using Visual Studio Code, the following shell scripts can be used to configure and build the project:

	./config
	./bld
	
## References

1. Maximilian Rest, W. Ross Dunkel, Kurt James Werner, Julius O. Smith, “RT-WDF—A MODULAR WAVE DIGITAL FILTER LIBRARY WITH SUPPORT FOR
ARBITRARY TOPOLOGIES AND MULTIPLE NONLINEARITIES,” in Proceedings of the 19th International Conference on Digital Audio Effects (DAFx-16), Brno, Czech Republic, September 5–9, 2016. [http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf](http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf)

