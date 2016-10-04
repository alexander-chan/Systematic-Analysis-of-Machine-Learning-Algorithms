#include <cv.h>
#include <ml.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

#define NUMBER_OF_ATTRIBUTES 10
#define NUMBER_OF_CLASSES 2

vector< vector<float> > cleanData(vector<string> &rawData, int dataType) {
	
	vector< vector<float> > parsedData;
	int rowCount;
	int columnSelect [NUMBER_OF_ATTRIBUTES];
	if (dataType == 1) {
		columnSelect[0] = 4;
		columnSelect[1] = 5;
		columnSelect[2] = 7;
		columnSelect[3] = 8;
		columnSelect[4] = 9;
		columnSelect[5] = 10;
		columnSelect[6] = 11;
		columnSelect[7] = 12;
		columnSelect[8] = 13;
		columnSelect[9] = 14;
	}
	else {
		columnSelect[0] = 2;
		columnSelect[1] = 3;
		columnSelect[2] = 5;
		columnSelect[3] = 6;
		columnSelect[4] = 7;
		columnSelect[5] = 8;
		columnSelect[6] = 9;
		columnSelect[7] = 10;
		columnSelect[8] = 11;
		columnSelect[9] = 12;
	}
	
	for(vector<string>::iterator it = rawData.begin(); it != rawData.end(); ++it) 
	{
		vector<float> row;
		istringstream ss(*it);
		string token;
		rowCount = 0;
		while(getline(ss, token, ',')) 
		{
			if((find(begin(columnSelect), end(columnSelect), rowCount)) != end(columnSelect)) {
					row.push_back(stof(token));
			}
			rowCount++;
		}
		parsedData.push_back(row);
	}
	
	return parsedData;
}

vector< vector<float> > readCSV(const string file)
{
	vector<string> buffer;
	ifstream configFile;
	configFile.exceptions(ifstream::badbit);
	try
	{
		configFile.open(file.c_str(),ifstream::in);
		if(configFile.is_open())
		{
			string line;
			while (getline(configFile,line))
			{
				buffer.push_back(line);
			}
			configFile.close();
		}           
	}
	catch (ifstream::failure e) {throw e;}
	
	istringstream reader(buffer.front());
	string line;
	getline(reader, line);
	buffer.erase(buffer.begin());
	
	string token;
	istringstream reader2(line);
	getline(reader2, token, ',');
	
	if (token == "timestampMs") return cleanData(buffer, 1);
	else return cleanData(buffer, 2);

}

Mat vectorToMAT(vector< vector<float> > &inVec)
{
 	int rows = inVec.size();
	int cols = inVec[0].size();
 
	Mat newMatrix(rows, cols, CV_32FC1);
	for (int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			newMatrix.at<float>(i,j) = inVec[i][j];
		}
	}
	return newMatrix;
}

int main( int argc, char** argv )
{
	
	Mat gameData;
	Mat idleData;
	
	vector<string> gameFiles;
	
	gameFiles.push_back("DrivingGame/eegIDRecord1-Driving.csv");
	gameFiles.push_back("DrivingGame/eegIDRecord2-Driving.csv");
	gameFiles.push_back("DrivingGame/eegIDRecord3-Driving.csv");
	gameFiles.push_back("DrivingGame/eegIDRecord4-Driving.csv");
	gameFiles.push_back("DrivingGame/eegIDRecord5-Driving.csv");
	gameFiles.push_back("DrivingGame/eegIDRecord6-Driving.csv");
	
	vector<string> idleFiles;
	
	idleFiles.push_back("Idle/eegData1-idle.csv");
	idleFiles.push_back("Idle/eegData2-idle.csv");
	idleFiles.push_back("Idle/eegData3-idle.csv");
	idleFiles.push_back("Idle/eegData4-idle.csv");
	idleFiles.push_back("Idle/eegData5-idle.csv");
	
	vector< vector<float> > rawData;
	vector< vector<float> > parsedData;
	
	for(int i = 0; i < static_cast<int>(gameFiles.size()); i++) {
		rawData = readCSV(gameFiles[i]);
		parsedData.insert(parsedData.end(), rawData.begin(), rawData.end());
	}
	
	gameData = vectorToMAT(parsedData);
	
	parsedData.clear();
	
	for(int i = 0; i < static_cast<int>(idleFiles.size()); i++) {
		rawData = readCSV(idleFiles[i]);
		parsedData.insert(parsedData.end(), rawData.begin(), rawData.end());
	}
	
	idleData = vectorToMAT(parsedData);
	
	int totalRows = idleData.rows + gameData.rows;
	
	cout << "Total Rows: " << totalRows << endl;
	
	Mat trainingData(totalRows/2, NUMBER_OF_ATTRIBUTES, CV_32FC1);
	Mat testingData(totalRows/2, NUMBER_OF_ATTRIBUTES, CV_32FC1);
	Mat trainingClasses(totalRows/2, 1, CV_32FC1);
	Mat testingClasses(totalRows/2, 1, CV_32FC1);
	RNG seed = RNG(-1);
	
	vector<int> idleSample;
	vector<int> gameSample;
	
	for(int i = 0; i < idleData.rows; i++)
	{
		idleSample.push_back(i);
	}
	
	randShuffle(idleSample);
	
	for(int i = 0; i < gameData.rows; i++)
	{
		gameSample.push_back(i);
	}
	
	cout << "Game Sample: " << gameSample.size() << endl;
	cout << "Idle Sample: " << idleSample.size() << endl;
	cout << "Game Rows: " << gameData.rows << endl;
	cout << "Idle Rows: " << idleData.rows << endl;
	cout << "Training: " << trainingData.rows << endl;
	cout << "Testing: " << testingData.rows << endl;
	
	randShuffle(gameSample);
	
	int i, j;
	
	for(i = 0; i < idleSample.size()/2; i++) {
		idleData.row(idleSample[i]).copyTo(trainingData.row(i));
		trainingClasses.at<float>(i) = 0.0;
	}
	cout << "Filled rows 0 - " << i << " of trainingData with idleData." << endl;
	
	const int save = i;	
	
	for(j = 0; j < idleSample.size()/2; j++) {
		idleData.row(idleSample[i]).copyTo(testingData.row(j));
		testingClasses.at<float>(j) = 0.0;
		i++;
	}
	cout << "Filled rows 0 - " << j << " of testingData with idleData." << endl;

	for(i = save; i < gameSample.size()/2 + save; i++) {
		gameData.row(gameSample[i - save]).copyTo(trainingData.row(i));
		trainingClasses.at<float>(i) = 1.0;
	}
	cout << "Filled rows " << save << " - " << i << " of trainingData with gameData." << endl;

	for(i = save; i < gameSample.size()/2 + save; i++) {
		gameData.row(gameSample[i - save]).copyTo(testingData.row(j));
		testingClasses.at<float>(j) = 1.0;
		j++;
	}
	cout << "Filled rows " << save << " - " << i << " of testingData with gameData." << endl;
	
	/*
	for(int i = 0; i < trainingData.rows; i++)
	{
		cout << trainingData.row(i) << endl;
	}
	*/
	
	FileStorage fs("data.xml", FileStorage::WRITE);
	
	fs << "trainingData" << trainingData;
	fs << "testingData" << testingData;
	fs << "trainingClasses" << trainingClasses;
	fs << "testingClasses" << testingClasses;
	fs << "gameFiles" << gameFiles;
	fs << "idleFiles" << idleFiles;
	
}
