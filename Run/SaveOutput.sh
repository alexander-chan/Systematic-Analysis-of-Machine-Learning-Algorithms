# This script runs the algorithms and saves all output in a file with an user-specified file name
# Assume in Run Directory
cd ../Output
echo "Write the filename (without suffix) of the output file: "
read input_variable
echo "The output file will be saved as $input_variable in the Output folder."

echo "Order of Output Data.">> $input_variable.txt
echo >> $input_variable.txt
echo "1. RandomForest">> $input_variable.txt
echo "2. Boosting">> $input_variable.txt
echo "3. Support Vector Machine (SVM)">> $input_variable.txt
echo "4. Bayes" >> $input_variable.txt
echo "5. K-Nearest Neighbors (KNN)">> $input_variable.txt
echo >> $input_variable.txt
echo "______________________________________" >> $input_variable.txt

echo "..."
cd ../RandomForest
./RandomForest >> ../Output/$input_variable.txt
echo "..."
cd ../Boosting
./Boosting >> ../Output/$input_variable.txt
echo "..."
cd ../SVM
./SVM >> ../Output/$input_variable.txt
echo "..."
cd ../Bayes
./Bayes >> ../Output/$input_variable.txt
echo "..."
cd ../KNN
./KNN >> ../Output/$input_variable.txt
echo "SaveOutput DONE"

