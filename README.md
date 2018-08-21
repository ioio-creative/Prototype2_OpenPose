## Added by Chris
This project makes use of the fantastic OpenPose project by CMU-Perceptual-Computing-Lab, which detects human skeleton points from 2D image feeds, such as images, videos and webcam.
https://github.com/CMU-Perceptual-Computing-Lab/openpose

To use this code with the OpenPose library, one has to build this code (a.k.a. user custom code) together with the library code from source. I have only tried building it using Visual Studio 2017 and Cuda SDK 9 on Windows 10.
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/user_code/README.md

This code imports the trained model data from OpenPose library, i.e. running the initialisation once, then repeartedly (in a while-true loop) read images (in order of create datetime) from INPUT directory, output the detected skeleton points from human body/bodies in a json text file (having the same name as the input image) in the OUTPUT directory. The input image read is then moved to the ARCHIVES directory. The process repeats indefinitely many times because it is in a while loop. The program will be running doing-nothing while-true loop when all the files in the INPUT directory are examined, until new images are added to the INPUT directory. That is, the program will not terminate automatically. 

The INPUT, OUTPUT and ARCHIVES directories are specified as command line arguments when starting the program.
Usage: BUILT_EXE input output archives

This code does not do sanity checs, such as checking whether the input image is indeed an image.

This code has the advantage that it only runs the trained model initialisation step once. Therefore, subsequent calls to the skeleton detection process using the trained model can be done very quickly.

This code does not make use of the command line flags predefined by OpenPose library because I did not have enough time / C++ knowledge to dig too deep into the library.
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md

One improvement to the code can be using network communication (e.g. OSC) to communicate the input and output data to / from a caller program. Currently, this code uses the OS file system as a medium to communicate input / output and the while-true loop will hold up the computing resources of one thread.

This code will be used with the Prototype2 project by the Research team of IOIO.
https://github.com/ioio-creative/Prototype2



Sections below are from the OpenPost library.

Adding and Testing Custom Code
====================================



## Purpose
You can quickly add your custom code into this folder so that quick prototypes can be easily tested without having to create a whoel new project just for it.



## How-to
1. Install/compile OpenPose as usual.
2. Add your custom *.cpp / *.hpp files here,. Hint: You might want to start by copying the [OpenPoseDemo](../openpose/openpose.cpp) example or any of the [examples/tutorial_wrapper/](../tutorial_wrapper/) examples. Then, you can simply modify their content.
3. Add the name of your custom *.cpp / *.hpp files at the top of the [examples/user_code/CMakeLists.txt](./CMakeLists.txt) file.
4. Re-compile OpenPose.
```
# Ubuntu/Mac
cd build/
make -j`nproc`
# Windows
# Close Visual Studio, re-run CMake, and re-compile the project in Visual Studio 
```
5. **Run step 4 every time that you make changes into your code**.



## Running your Custom Code
Run:
```
./build/examples/user_code/{your_custom_file_name}
```
