#include "JTM45.cpp"
#include <cmath>
#include <chrono>
#include <catch/catch.hpp>
#include "AudioFile.h"
// apt install libgoogle-perftools-dev

using namespace std::chrono;


extern void StartTiming();
extern void TimingReport();

extern bool g_UseInterpolatedTriode;



TEST_CASE( "JTM45 Performance", "[performance]" ) 
{

    g_UseInterpolatedTriode = true;
    LV2_Handle hJtm = JTM45::instantiate(nullptr,44100,nullptr,nullptr);
    JTM45*pJtm45 = (JTM45*)hJtm;


    double dx = M_PI *2 * 4000/48000.0;
    double x = 0;

    float inputBuffer[64];
    float outputBuffer[64];
    memset(inputBuffer,0,sizeof(inputBuffer));
    memset(outputBuffer,0,sizeof(inputBuffer));

    float gain = 0.8f;
    float volume = 0.1f;

    JTM45::connect_port(hJtm,IN,inputBuffer);
    JTM45::connect_port(hJtm,OUT_1,outputBuffer);
    JTM45::connect_port(hJtm,GAIN,&gain);
    JTM45::connect_port(hJtm,VOLUME,&volume);

    JTM45::activate(hJtm);
    cout << "Generating 5 seconds of audio..." << endl;

    //ProfilerStart("jtm45test.prof");
    StartTiming();

    for (uint64_t t = 0; t < 48000*5; t += 64)
    {
        for (int i = 0; i < 64; ++i)
        {
            inputBuffer[i] = 10*(float)std::sin(x);
            x += dx;
        }
        
        JTM45::run(hJtm,64);
    }
    //ProfilerStop();

    TimingReport();

}





TEST_CASE( "JTM45 Write Audio", "[writeAudio]" ) 
{

    g_UseInterpolatedTriode = true;
    LV2_Handle hJtm = JTM45::instantiate(nullptr,44100,nullptr,nullptr);
    JTM45*pJtm45 = (JTM45*)hJtm;


    double dx = M_PI *2 * 440/48000.0;
    double x = 0;

    float inputBuffer[64];
    float outputBuffer[64];
    memset(inputBuffer,0,sizeof(inputBuffer));
    memset(outputBuffer,0,sizeof(inputBuffer));

    float gain = 1.0f;
    float volume = 0.1f;

    JTM45::connect_port(hJtm,IN,inputBuffer);
    JTM45::connect_port(hJtm,OUT_1,outputBuffer);
    JTM45::connect_port(hJtm,GAIN,&gain);
    JTM45::connect_port(hJtm,VOLUME,&volume);

    AudioFile<float> a;
    a.setNumChannels (1);
    a.setNumSamplesPerChannel (48000);


    JTM45::activate(hJtm);
    cout << "Writing 5 seconds of audio..." << endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (uint64_t t = 0; t < 48000*5; t += 64)
    {
        for (int i = 0; i < 64; ++i)
        {
            inputBuffer[i] = 6.8*(float)std::sin(x);
            x += dx;
        }
        
        JTM45::run(hJtm,64);

        for (int i = 0; i < 64; ++i)
        {
            a.samples[0].push_back(outputBuffer[i]);
        }

    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::stringstream ss;
    ss << getenv("HOME") << "/tmp/testWave.wav";
    std::string fileName = ss.str();
    a.save(fileName,AudioFileFormat::Wave);

}