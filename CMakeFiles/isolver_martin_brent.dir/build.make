# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/Documents/General_electrochemistry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/Documents/General_electrochemistry

# Include any dependencies generated for this target.
include CMakeFiles/isolver_martin_brent.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/isolver_martin_brent.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/isolver_martin_brent.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/isolver_martin_brent.dir/flags.make

CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o: CMakeFiles/isolver_martin_brent.dir/flags.make
CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o: src/surface_process_brent.cpp
CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o: CMakeFiles/isolver_martin_brent.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user/Documents/General_electrochemistry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o -MF CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o.d -o CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o -c /home/user/Documents/General_electrochemistry/src/surface_process_brent.cpp

CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/Documents/General_electrochemistry/src/surface_process_brent.cpp > CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.i

CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/Documents/General_electrochemistry/src/surface_process_brent.cpp -o CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.s

# Object files for target isolver_martin_brent
isolver_martin_brent_OBJECTS = \
"CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o"

# External object files for target isolver_martin_brent
isolver_martin_brent_EXTERNAL_OBJECTS =

isolver_martin_brent.cpython-310-x86_64-linux-gnu.so: CMakeFiles/isolver_martin_brent.dir/src/surface_process_brent.cpp.o
isolver_martin_brent.cpython-310-x86_64-linux-gnu.so: CMakeFiles/isolver_martin_brent.dir/build.make
isolver_martin_brent.cpython-310-x86_64-linux-gnu.so: CMakeFiles/isolver_martin_brent.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/user/Documents/General_electrochemistry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module isolver_martin_brent.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/isolver_martin_brent.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/user/Documents/General_electrochemistry/isolver_martin_brent.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/isolver_martin_brent.dir/build: isolver_martin_brent.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/isolver_martin_brent.dir/build

CMakeFiles/isolver_martin_brent.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/isolver_martin_brent.dir/cmake_clean.cmake
.PHONY : CMakeFiles/isolver_martin_brent.dir/clean

CMakeFiles/isolver_martin_brent.dir/depend:
	cd /home/user/Documents/General_electrochemistry && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/Documents/General_electrochemistry /home/user/Documents/General_electrochemistry /home/user/Documents/General_electrochemistry /home/user/Documents/General_electrochemistry /home/user/Documents/General_electrochemistry/CMakeFiles/isolver_martin_brent.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/isolver_martin_brent.dir/depend

