# builds all the algorithms (RandomForest, SVM, ...) in the case when the code is changed
# Assume in Run folder
cd ../RandomForest
rm CMakeCache.txt
cmake CMakeLists.txt
make
cd ../Boosting
rm CMakeCache.txt
cmake CMakeLists.txt
make
cd ../SVM
rm CMakeCache.txt
cmake CMakeLists.txt
make
cd ../KNN
rm CMakeCache.txt
cmake CMakeLists.txt
make
cd ../Bayes
rm CMakeCache.txt
cmake CMakeLists.txt
make
cd ../Run
echo "Build Algorithms DONE"

