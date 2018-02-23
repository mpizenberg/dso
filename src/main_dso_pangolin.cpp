/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include <boost/thread.hpp>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"

#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
bool useSampleOutput = false;

int mode = 0;

bool firstRosSpin = false;

using namespace dso;

void my_exit_handler(int s) {
  printf("Caught signal %d\n", s);
  exit(1);
}

void exitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  firstRosSpin = true;
  while (true)
    pause();
}

void settingsDefault(int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0) {
    printf("DEFAULT settings:\n"
           "- %s real-time enforcing\n"
           "- 2000 active points\n"
           "- 5-7 active frames\n"
           "- 1-6 LM iteration each KF\n"
           "- original image resolution\n",
           preset == 0 ? "no " : "1x");

    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    setting_logStuff = false;
  }

  if (preset == 2) {
    printf("FAST settings:\n"
           "- %s real-time enforcing\n"
           "- 800 active points\n"
           "- 4-6 active frames\n"
           "- 1-4 LM iteration each KF\n"
           "- 424 x 320 image resolution\n",
           preset == 0 ? "no " : "5x");

    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

void parseArgument(char *arg) {
  int option;
  float foption;
  char buf[1000];

  if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
    if (option == 1) {
      useSampleOutput = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "quiet=%d", &option)) {
    if (option == 1) {
      setting_debugout_runquiet = true;
      printf("QUIET MODE, I'll shut up!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "preset=%d", &option)) {
    settingsDefault(option);
    return;
  }

  if (1 == sscanf(arg, "rec=%d", &option)) {
    if (option == 0) {
      disableReconfigure = true;
      printf("DISABLE RECONFIGURE!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "noros=%d", &option)) {
    if (option == 1) {
      disableROS = true;
      disableReconfigure = true;
      printf("DISABLE ROS (AND RECONFIGURE)!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option)) {
    if (option == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nogui=%d", &option)) {
    if (option == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option)) {
    if (option == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "prefetch=%d", &option)) {
    if (option == 1) {
      prefetch = true;
      printf("PREFETCH!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "start=%d", &option)) {
    start = option;
    printf("START AT %d!\n", start);
    return;
  }
  if (1 == sscanf(arg, "end=%d", &option)) {
    end = option;
    printf("END AT %d!\n", start);
    return;
  }

  if (1 == sscanf(arg, "files=%s", buf)) {
    source = buf;
    printf("loading data from %s!\n", source.c_str());
    return;
  }

  if (1 == sscanf(arg, "calib=%s", buf)) {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }

  if (1 == sscanf(arg, "vignette=%s", buf)) {
    vignette = buf;
    printf("loading vignette from %s!\n", vignette.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    gammaCalib = buf;
    printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
    return;
  }

  if (1 == sscanf(arg, "rescale=%f", &foption)) {
    rescale = foption;
    printf("RESCALE %f!\n", rescale);
    return;
  }

  if (1 == sscanf(arg, "save=%d", &option)) {
    if (option == 1) {
      debugSaveImages = true;
      if (42 == system("rm -rf images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("rm -rf images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option)) {

    mode = option;
    if (option == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char **argv) {
  // setlocale(LC_ALL, "");
  for (int i = 1; i < argc; i++)
    parseArgument(argv[i]);

  // hook crtl+C.
  boost::thread exThread = boost::thread(exitThread);

  ImageFolderReader *reader =
      new ImageFolderReader(source, calib, gammaCalib, vignette);
  reader->setGlobalCalibration();
  end = reader->getNumImages();

  if (setting_photometricCalibration > 0 &&
      reader->getPhotometricGamma() == 0) {
    printf("ERROR: dont't have photometric calibation. Need to use commandline "
           "options mode=1 or mode=2 ");
    exit(1);
  }

  FullSystem *fullSystem = new FullSystem();
  fullSystem->setGammaFunction(reader->getPhotometricGamma());

  IOWrap::PangolinDSOViewer *viewer = 0;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
    fullSystem->outputWrapper.push_back(viewer);
  }

  if (useSampleOutput)
    fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

  // to make MacOS happy: run this in dedicated thread -- and use this one to
  // run the GUI.
  std::thread runthread([&]() {

    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();

    for (int i = start; i < end; i++) {
      if (!fullSystem->initialized) { // if not initialized: reset start time.
        gettimeofday(&tv_start, NULL);
        started = clock();
      }

      ImageAndExposure *img;
      img = reader->getImage(i);
      fullSystem->addActiveFrame(img, i);
      delete img;

      if (fullSystem->initFailed || setting_fullResetRequested) {
        printf("RESETTING!\n");
        std::vector<IOWrap::Output3DWrapper *> wraps =
            fullSystem->outputWrapper;
        delete fullSystem;

        for (IOWrap::Output3DWrapper *ow : wraps)
          ow->reset();

        fullSystem = new FullSystem();
        fullSystem->setGammaFunction(reader->getPhotometricGamma());
        fullSystem->linearizeOperation = true;
        fullSystem->outputWrapper = wraps;
        setting_fullResetRequested = false;
      }

      if (fullSystem->isLost) {
        printf("LOST!!\n");
        break;
      }
    }
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    fullSystem->printResult("result.txt");

    int numFramesProcessed = end - start;
    double numSecondsProcessed =
        reader->getTimestamp(end - 1) - reader->getTimestamp(start);
    double MilliSecondsTakenSingle =
        1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
    printf("\n======================"
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame (single core); "
           "\n%.3fx (single core); "
           "\n======================\n\n",
           numFramesProcessed, numFramesProcessed / numSecondsProcessed,
           MilliSecondsTakenSingle / numFramesProcessed,
           1000 / (MilliSecondsTakenSingle / numSecondsProcessed));

  });

  if (viewer != 0)
    viewer->run();

  runthread.join();

  for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
    ow->join();
    delete ow;
  }

  printf("DELETE FULLSYSTEM!\n");
  delete fullSystem;

  printf("DELETE READER!\n");
  delete reader;

  printf("EXIT NOW!\n");
  return 0;
}
