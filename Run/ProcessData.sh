# This script builds and runs the Process Data code, which gathers data from csv files and outputs a data.xml file that openCV can analyze
# Assume inside Run Directory
cd ../Data
rm CMakeCache.txt
cmake CMakeLists.txt
make
./dataProcess
echo "Process Data Done"
cd .. #return back to UHDResearch Directory
