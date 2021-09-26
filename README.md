# jtm45

A port of the JTM45 Amp Emulator LV2 Plugin, highly optimized for Raspberry Pi 4/aarch64. Runs about 23x faster
than the original code. 

Just barely runs in realtime on a Raspberry Pi 4 B, running on a 64-bit platform. You may need to 
boost your CPU speed to 1750MHz to get it to run. You may need to run on a realtime kernel.

Tested on Ubuntu Studio for ARM 64-bit hosted by PiPedal. 


Based on https://github.com/MaxPayne86/rt-wdf_lv2

    *rt-wdf_lv2*


    Implementation of LV2 audio plugins using rt-wdf library

    Documentation on rt-wdf library can be found [here](http://dafx16.vutbr.cz/dafxpapers/40-DAFx-16_paper_35-PN.pdf)

    LV2 plugins:

    - JTM45
