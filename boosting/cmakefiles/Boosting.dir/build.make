# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vm642/Desktop/UHDResearch/Boosting

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vm642/Desktop/UHDResearch/Boosting

# Include any dependencies generated for this target.
include CMakeFiles/Boosting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Boosting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Boosting.dir/flags.make

CMakeFiles/Boosting.dir/Boosting.cpp.o: CMakeFiles/Boosting.dir/flags.make
CMakeFiles/Boosting.dir/Boosting.cpp.o: Boosting.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vm642/Desktop/UHDResearch/Boosting/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Boosting.dir/Boosting.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Boosting.dir/Boosting.cpp.o -c /home/vm642/Desktop/UHDResearch/Boosting/Boosting.cpp

CMakeFiles/Boosting.dir/Boosting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Boosting.dir/Boosting.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vm642/Desktop/UHDResearch/Boosting/Boosting.cpp > CMakeFiles/Boosting.dir/Boosting.cpp.i

CMakeFiles/Boosting.dir/Boosting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Boosting.dir/Boosting.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vm642/Desktop/UHDResearch/Boosting/Boosting.cpp -o CMakeFiles/Boosting.dir/Boosting.cpp.s

CMakeFiles/Boosting.dir/Boosting.cpp.o.requires:
.PHONY : CMakeFiles/Boosting.dir/Boosting.cpp.o.requires

CMakeFiles/Boosting.dir/Boosting.cpp.o.provides: CMakeFiles/Boosting.dir/Boosting.cpp.o.requires
	$(MAKE) -f CMakeFiles/Boosting.dir/build.make CMakeFiles/Boosting.dir/Boosting.cpp.o.provides.build
.PHONY : CMakeFiles/Boosting.dir/Boosting.cpp.o.provides

CMakeFiles/Boosting.dir/Boosting.cpp.o.provides.build: CMakeFiles/Boosting.dir/Boosting.cpp.o

# Object files for target Boosting
Boosting_OBJECTS = \
"CMakeFiles/Boosting.dir/Boosting.cpp.o"

# External object files for target Boosting
Boosting_EXTERNAL_OBJECTS =

Boosting: CMakeFiles/Boosting.dir/Boosting.cpp.o
Boosting: CMakeFiles/Boosting.dir/build.make
Boosting: /usr/local/lib/libopencv_videostab.so.2.4.9
Boosting: /usr/local/lib/libopencv_video.so.2.4.9
Boosting: /usr/local/lib/libopencv_ts.a
Boosting: /usr/local/lib/libopencv_superres.so.2.4.9
Boosting: /usr/local/lib/libopencv_stitching.so.2.4.9
Boosting: /usr/local/lib/libopencv_photo.so.2.4.9
Boosting: /usr/local/lib/libopencv_ocl.so.2.4.9
Boosting: /usr/local/lib/libopencv_objdetect.so.2.4.9
Boosting: /usr/local/lib/libopencv_nonfree.so.2.4.9
Boosting: /usr/local/lib/libopencv_ml.so.2.4.9
Boosting: /usr/local/lib/libopencv_legacy.so.2.4.9
Boosting: /usr/local/lib/libopencv_imgproc.so.2.4.9
Boosting: /usr/local/lib/libopencv_highgui.so.2.4.9
Boosting: /usr/local/lib/libopencv_gpu.so.2.4.9
Boosting: /usr/local/lib/libopencv_flann.so.2.4.9
Boosting: /usr/local/lib/libopencv_features2d.so.2.4.9
Boosting: /usr/local/lib/libopencv_core.so.2.4.9
Boosting: /usr/local/lib/libopencv_contrib.so.2.4.9
Boosting: /usr/local/lib/libopencv_calib3d.so.2.4.9
Boosting: /usr/local/lib/libopencv_nonfree.so.2.4.9
Boosting: /usr/local/lib/libopencv_ocl.so.2.4.9
Boosting: /usr/local/lib/libopencv_gpu.so.2.4.9
Boosting: /usr/local/lib/libopencv_photo.so.2.4.9
Boosting: /usr/local/lib/libopencv_objdetect.so.2.4.9
Boosting: /usr/local/lib/libopencv_legacy.so.2.4.9
Boosting: /usr/local/lib/libopencv_video.so.2.4.9
Boosting: /usr/local/lib/libopencv_ml.so.2.4.9
Boosting: /usr/local/lib/libopencv_calib3d.so.2.4.9
Boosting: /usr/local/lib/libopencv_features2d.so.2.4.9
Boosting: /usr/local/lib/libopencv_highgui.so.2.4.9
Boosting: /usr/local/lib/libopencv_imgproc.so.2.4.9
Boosting: /usr/local/lib/libopencv_flann.so.2.4.9
Boosting: /usr/local/lib/libopencv_core.so.2.4.9
Boosting: CMakeFiles/Boosting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Boosting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Boosting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Boosting.dir/build: Boosting
.PHONY : CMakeFiles/Boosting.dir/build

CMakeFiles/Boosting.dir/requires: CMakeFiles/Boosting.dir/Boosting.cpp.o.requires
.PHONY : CMakeFiles/Boosting.dir/requires

CMakeFiles/Boosting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Boosting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Boosting.dir/clean

CMakeFiles/Boosting.dir/depend:
	cd /home/vm642/Desktop/UHDResearch/Boosting && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vm642/Desktop/UHDResearch/Boosting /home/vm642/Desktop/UHDResearch/Boosting /home/vm642/Desktop/UHDResearch/Boosting /home/vm642/Desktop/UHDResearch/Boosting /home/vm642/Desktop/UHDResearch/Boosting/CMakeFiles/Boosting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Boosting.dir/depend
