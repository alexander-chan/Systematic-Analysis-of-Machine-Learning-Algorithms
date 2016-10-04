#include <cv.h>
#include <ml.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include "../RawData/fileNames/mathFiles.h"
#include "../RawData/fileNames/drivingFiles.h"

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

	bool train;

	if(argc < 2) {
		printf("Please specify '--test' or '--train'\nExiting...\n");
		exit(0);
	}
	else {
		if(strcmp(argv[1], "--test") == 0) {
			train = false;
		}
		else if(strcmp(argv[1], "--train") == 0) {
			train = true;
		}
		else {
			printf("Please specify '--test' or '--train'\nExiting...\n");
			exit(0);
		}
	}

	Mat gameData;
	Mat idleData;
	
	vector<string> testFiles;
	
	vector<string> gameFiles;
	
	
	gameFiles.push_back(math::g10);
	gameFiles.push_back(math::g11);
	gameFiles.push_back(math::g12);
	gameFiles.push_back(math::g13);
	//gameFiles.push_back(math::g01);
	//gameFiles.push_back(math::g01);
	//gameFiles.push_back(math::g01);

	vector<string> idleFiles;
	
	idleFiles.push_back(math::i11);
	idleFiles.push_back(math::i12);
	idleFiles.push_back(math::i13);
	//idleFiles.push_back(math::i02);
	//idleFiles.push_back(math::i02);
				
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

	cout << "Game Rows: " << gameData.rows << endl;
	cout << "Idle Rows: " << idleData.rows << endl;

	cout << "What percentage of the data would you like to use? ";

	string buffer;
 	getline (cin,buffer);

 	int percentage;
 	stringstream(buffer) >> percentage;

	vector<int> idleSample;
	vector<int> gameSample;

	int idleRows = idleData.rows * percentage/100;
	int gameRows = gameData.rows * percentage/100;

	totalRows = idleRows + gameRows;

	cout << "Used Game Rows: " << gameRows << endl;
	cout << "Used Idle Rows: " << idleRows << endl;

	for(int i = 0; i < idleRows; i++)
	{
		idleSample.push_back(i);
	}

	RNG seed = RNG(-1);
	randShuffle(idleSample);
	
	for(int i = 0; i < gameRows; i++)
	{
		gameSample.push_back(i);
	}

	seed = RNG(27);
	randShuffle(gameSample);

	cout << "Please enter a filename to save as: ";

	string filename;

 	getline (cin,buffer);
  	stringstream(buffer) >> filename;

  	filename = filename + ".xml";

	FileStorage fs(filename, FileStorage::WRITE);

	if(train) {
		Mat trainingData(totalRows, NUMBER_OF_ATTRIBUTES, CV_32FC1);
		Mat trainingClasses(totalRows, 1, CV_32FC1);
		int i, j;
		for(i = 0; i < idleRows; i++) {
			idleData.row(idleSample[i]).copyTo(trainingData.row(i));
			trainingClasses.at<float>(i) = 0;
		}
		for(j = 0; j < gameRows; j++) {
			gameData.row(gameSample[j]).copyTo(trainingData.row(i));
			trainingClasses.at<float>(i) = 1;
			i++;
		}
		fs << "trainingData" << trainingData;
		fs << "trainingClasses" << trainingClasses;
		/*
		for(i = 0; i < trainingData.rows; i++)
		{
			cout << trainingData.row(i) << endl;
		}
		*/
	}
	else {
		Mat testingData(totalRows, NUMBER_OF_ATTRIBUTES, CV_32FC1);
		Mat testingClasses(totalRows, 1, CV_32FC1);
		int i, j;
		for(i = 0; i < idleRows; i++) {
			idleData.row(idleSample[i]).copyTo(testingData.row(i));
			testingClasses.at<float>(i) = 0;
		}
		for(j = 0; j < gameRows; j++) {
			gameData.row(gameSample[j]).copyTo(testingData.row(i));
			testingClasses.at<float>(i) = 1;
			i++;
		}
		
		fs << "testingData" << testingData;
		fs << "testingClasses" << testingClasses;
		
		/*
		for(i = 0; i < testingData.rows; i++)
		{
			cout << testingData.row(i) << endl;
		}
		*/
	}
	
	cout << "Data saved as '" << filename << "'" << endl;
}
