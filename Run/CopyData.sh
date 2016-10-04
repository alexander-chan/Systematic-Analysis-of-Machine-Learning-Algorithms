# This code copies the data.xml file from the Process data folder and into the folders of each algorithm
# Assume inside Run Directory
cd ../Data
cp ./data.xml ../RandomForest
cp ./data.xml ../Boosting
cp ./data.xml ../KNN
cp ./data.xml ../SVM
cp ./data.xml ../Bayes
echo "Copy Data Done"
cd .. #return back to UHDResearch Directory
