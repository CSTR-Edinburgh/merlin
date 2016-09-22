//-----------------------------------------------------------------------------
// 
// Author: Zhizheng Wu (wuzhizheng@gmail.com)
// Date: 11-03-2016
//
// To generate waveform given F0, band aperiodicities and spectrum with WORLD vocoder
//
// This is modified based on Msanori Morise's test.cpp. Low-dimensional band aperiodicities are used as suggested by Oliver.
//
// synth FFT_length sampling_rate F0_file spectrogram_file aperiodicity_file output_waveform
//
//-----------------------------------------------------------------------------

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <sys/stat.h> 

#if (defined (__WIN32__) || defined (_WIN32)) && !defined (__MINGW32__)
#include <conio.h>
#include <windows.h>
#pragma comment(lib, "winmm.lib")
#pragma warning(disable : 4996)
#endif
#if (defined (__linux__) || defined(__CYGWIN__) || defined(__APPLE__))
#include <stdint.h>
#include <sys/time.h>
#endif

// For .wav input/output functions.
#include "audioio.h"

// WORLD core functions.
// Note: win.sln uses an option in Additional Include Directories.
// To compile the program, the option "-I $(SolutionDir)..\src" was set.
#include "world/matlabfunctions.h"
#include "world/synthesis.h"
#include "world/common.h"
#include "world/constantnumbers.h"

// Frame shift [msec]
#define FRAMEPERIOD 5.0

#if (defined (__linux__) || defined(__CYGWIN__) || defined(__APPLE__))
// Linux porting section: implement timeGetTime() by gettimeofday(),
#ifndef DWORD
#define DWORD uint32_t
#endif
DWORD timeGetTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  DWORD ret = static_cast<DWORD>(tv.tv_usec / 1000 + tv.tv_sec * 1000);
  return ret;
}
#endif

//-----------------------------------------------------------------------------
// struct for WORLD
// This struct is an option.
// Users are NOT forced to use this struct.
//-----------------------------------------------------------------------------
typedef struct {
  double frame_period;
  int fs;

  double *f0;
  double *time_axis;
  int f0_length;

  double **spectrogram;
  double **aperiodicity;
  int fft_size;
} WorldParameters;

namespace {

void DisplayInformation(int fs, int nbit, int x_length) {
  printf("File information\n");
  printf("Sampling : %d Hz %d Bit\n", fs, nbit);
  printf("Length %d [sample]\n", x_length);
  printf("Length %f [sec]\n", static_cast<double>(x_length) / fs);
}

void WaveformSynthesis(WorldParameters *world_parameters, int fs,
    int y_length, double *y) {
  DWORD elapsed_time;
  // Synthesis by the aperiodicity
//  printf("\nSynthesis\n");
//  elapsed_time = timeGetTime();
  Synthesis(world_parameters->f0, world_parameters->f0_length,
      world_parameters->spectrogram, world_parameters->aperiodicity,
      world_parameters->fft_size, world_parameters->frame_period, fs,
      y_length, y);
//  printf("WORLD: %d [msec]\n", timeGetTime() - elapsed_time);
}

void DestroyMemory(WorldParameters *world_parameters) {
  delete[] world_parameters->time_axis;
  delete[] world_parameters->f0;
  for (int i = 0; i < world_parameters->f0_length; ++i) {
    delete[] world_parameters->spectrogram[i];
    delete[] world_parameters->aperiodicity[i];
  }
  delete[] world_parameters->spectrogram;
  delete[] world_parameters->aperiodicity;
}

}  // namespace

//-----------------------------------------------------------------------------
// Test program.
// test.exe input.wav outout.wav f0 spec flag
// input.wav  : argv[1] Input file
// output.wav : argv[2] Output file
// f0         : argv[3] F0 scaling (a positive number)
// spec       : argv[4] Formant shift (a positive number)
//-----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf("command: synth FFT_length sampling_rate F0_file spectrogram_file aperiodicity_file output_waveform\n");
    return -2;
  }

  int fft_size = atoi(argv[1]);
  int fs = atoi(argv[2]);


  // compute n bands from fs as in d4c.cpp:325   
  int number_of_aperiodicities = static_cast<int>(MyMinDouble(world::kUpperLimit, fs / 2.0 -
      world::kFrequencyInterval) / world::kFrequencyInterval);

  WorldParameters world_parameters = { 0 };
  // You must set fs and frame_period before analysis/synthesis.
  world_parameters.fs = fs;

  // 5.0 ms is the default value.
  // Generally, the inverse of the lowest F0 of speech is the best.
  // However, the more elapsed time is required.
  world_parameters.frame_period = 5.0;
  world_parameters.fft_size = fft_size;


  // find number of frames (doubles) in f0 file:  
  struct stat st;
  if (stat(argv[3], &st) == -1) {
    printf("cannot read f0\n");
    return -2;
    }
  int f0_length = (st.st_size / sizeof(double));
  world_parameters.f0_length = f0_length;

//  printf("%d\n", f0_length);
  
  world_parameters.f0 = new double[f0_length];

  FILE *fp;
  fp = fopen(argv[3], "rb");
  for (int i = 0; i < f0_length; i++) {
	fread(&world_parameters.f0[i], sizeof(double), 1, fp);
  }
  fclose(fp);

  double **coarse_aperiodicities = new double *[world_parameters.f0_length];
  world_parameters.aperiodicity = new double *[world_parameters.f0_length];
  for (int i = 0; i < world_parameters.f0_length; ++i) {
    world_parameters.aperiodicity[i] = new double[fft_size / 2 + 1];
    coarse_aperiodicities[i]  = new double[number_of_aperiodicities];
  }

  world_parameters.spectrogram = new double *[world_parameters.f0_length];
  for (int i = 0; i < world_parameters.f0_length; ++i) {
    world_parameters.spectrogram[i] = new double[fft_size / 2 + 1];
  }

  fp = fopen(argv[4], "rb");
  for (int i = 0; i < f0_length; i++) {
    for (int j = 0; j < fft_size / 2 + 1; j++) {
      fread(&world_parameters.spectrogram[i][j], sizeof(double), 1, fp);
    }
  }
  fclose(fp);

  // aper
  fp = fopen(argv[5], "rb");
  for (int i = 0; i < f0_length; i++) {
    for (int j = 0; j < number_of_aperiodicities; j++) {
      fread(&coarse_aperiodicities[i][j], sizeof(double), 1, fp);
    }
  }
  fclose(fp);  



  // convert bandaps to full aperiodic spectrum by interpolation (originally in d4c extraction):
  
  // Linear interpolation to convert the coarse aperiodicity into its
  // spectral representation.
  
  // -- for interpolating --
  double *coarse_aperiodicity = new double[number_of_aperiodicities + 2];
  coarse_aperiodicity[0] = -60.0;
  coarse_aperiodicity[number_of_aperiodicities + 1] = 0.0;
  double *coarse_frequency_axis = new double[number_of_aperiodicities + 2];
  for (int i = 0; i <= number_of_aperiodicities; ++i)
    coarse_frequency_axis[i] =
      static_cast<double>(i) * world::kFrequencyInterval;
  coarse_frequency_axis[number_of_aperiodicities + 1] = fs / 2.0;

  double *frequency_axis = new double[fft_size / 2 + 1];
  for (int i = 0; i <= fft_size / 2; ++i)
    frequency_axis[i] = static_cast<double>(i) * fs / fft_size;
  // ----
  
  for (int i = 0; i < f0_length; ++i) {
    // load band ap values for this frame into  coarse_aperiodicity
    for (int k = 0; k < number_of_aperiodicities; ++k) {
        coarse_aperiodicity[k+1] = coarse_aperiodicities[i][k];
    }
    interp1(coarse_frequency_axis, coarse_aperiodicity,
      number_of_aperiodicities + 2, frequency_axis, fft_size / 2 + 1,
      world_parameters.aperiodicity[i]);
    for (int j = 0; j <= fft_size / 2; ++j)
      world_parameters.aperiodicity[i][j] = pow(10.0, world_parameters.aperiodicity[i][j] / 20.0);
  }  
  
  
  //printf("%d %d\n", world_parameters.f0_length, fs);
  
  //---------------------------------------------------------------------------
  // Synthesis part
  //---------------------------------------------------------------------------
  // The length of the output waveform
  int y_length = static_cast<int>((world_parameters.f0_length - 1) *
    FRAMEPERIOD / 1000.0 * fs) + 1;
  double *y = new double[y_length];
  // Synthesis
  WaveformSynthesis(&world_parameters, fs, y_length, y);

  // Output
  wavwrite(y, y_length, fs, 16, argv[6]);

  delete[] y;
  DestroyMemory(&world_parameters);

  for (int i=0; i<f0_length; i++){
    delete[] coarse_aperiodicities[i];
  }
  delete[] coarse_aperiodicities;
  delete[] coarse_aperiodicity;
  delete[] frequency_axis;
    
  printf("complete %s.\n", argv[6]);
  return 0;
}
