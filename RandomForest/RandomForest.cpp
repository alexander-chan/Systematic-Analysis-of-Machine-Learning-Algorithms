#include <cv.h>
#include <ml.h>
#include <stdio.h>

using namespace cv;

#define NUMBER_OF_ATTRIBUTES 10
#define NUMBER_OF_CLASSES 2

int main( int argc, char** argv )
{
	bool verbose = false;
	bool train = false;
	bool test = false;
	char * testFile;
	char * trainFile;
	char * modelFile;

	if(argc > 1) {
		for(int i = 1; i < argc; i++) {
			if(strcmp(argv[i], "-v") == 0) {
				verbose = true;
			}
			else if(strcmp(argv[i], "-t") == 0) {
				test = true;
				if(i == argc - 1) {
					std::cout << "A test file name must be provided when using the '-t' option.\nExiting...\n" << std::endl;
					exit(0);
				}
				testFile = argv[i+1];
				printf("Using testing data from '%s'\n", testFile);
				i++;
			}
			else if(strcmp(argv[i], "-r") == 0) {
				train = true;
				if(i == argc - 1) {
					std::cout << "A training file name must be provided when using the '-r' option.\nExiting...\n" << std::endl;
					exit(0);
				}
				trainFile = argv[i+1];
				printf("Using training data from '%s'\n", trainFile);
				i++;
			}
			else if(strcmp(argv[i], "-o") == 0) {
				if(i == argc - 1) {
					std::cout << "A model file name must be provided when using the '-o' option.\nExiting...\n" << std::endl;
					exit(0);
				}
				modelFile = argv[i+1];
				printf("Using pre-trained model from '%s'\n", modelFile);
				i++;
			}
		}
		if(test && train) {
			printf("Cannot test and train at the same time.\nExiting...\n");
			exit(0);
		}
		else if(!(test || train)) {
			printf("Please use either '-t' to test or '-r' to train.\nExiting...\n");
			exit(0);
		}
	}
	else {
		printf("Please use '-t' to test, '-r' to train, or '-o' to load a model.\nExiting...\n");
		exit(0);
	}
	
	FileStorage fs;
	
	printf ("Using OpenCV version %s\n\n", CV_VERSION);
	
	clock_t t1,t2,t3;
	t1=clock();
	
	CvRTrees* randomForest = new CvRTrees;
	
	if(train) {
		
		if(!fs.open(trainFile, FileStorage::READ)) {
			printf("Could not locate data file '%s'\nExiting...\n", trainFile);
			exit(0);
		};
		
		Mat trainingData;
		Mat trainingClasses;
		
		fs["trainingData"] >> trainingData;
		fs["trainingClasses"] >> trainingClasses;
	
		int trainingRows = trainingData.rows;

		Mat varType = Mat(NUMBER_OF_ATTRIBUTES + 1, 1, CV_8U );
		varType.setTo(Scalar(CV_VAR_NUMERICAL) );

		varType.at<uchar>(NUMBER_OF_ATTRIBUTES, 0) = CV_VAR_CATEGORICAL;

		float priors[] = {1,1};

		CvRTParams params = CvRTParams(25, 	// maximum depth
						    5, 		// minimum sample count
						    0, 		// regression accuracy
						    false, 	// compute surrogate split, no missing data
						    15, 		// max categories
						    priors, 	// priors
						    true,  	// calculate variable importance
						    5,       	// size of subset
						    100,	 	// max number of trees in the forest
						    0.01f,	// forest accuracy
						    CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination criteria
						    );
		

		randomForest->train(trainingData, CV_ROW_SAMPLE, trainingClasses, Mat(), Mat(), varType, Mat(), params);

		randomForest->save("trainedForest.xml", "randomForest");
		
		printf("Model was trained and saved as 'trainedForest.xml'\n");
	}
	else {

		if(!fs.open(testFile, FileStorage::READ)) {
			printf("Could not locate data file '%s'\nExiting...\n", testFile);
			exit(0);
		};
		
		randomForest->load(modelFile, "randomForest");

		Mat testingData;
		Mat testingClasses;
		
		fs["testingData"] >> testingData;
		fs["testingClasses"] >> testingClasses;
	
		int testingRows = testingData.rows;

		double result;

		Mat testSample;
	
		int confusion[2][2] = {0,0,0,0};

		for (int tsample = 0; tsample < testingRows; tsample++)
		{

			testSample = testingData.row(tsample);

			result = randomForest->predict(testSample, Mat());

			//printf("Sample %i -> class is (%d)\n", tsample, (int) result);
		
			if(result == 1)
			{
				if(testingClasses.at<float>(tsample) == 1) confusion[0][0]++;
				else confusion[1][0]++;
			}
			else
			{
				if(testingClasses.at<float>(tsample) != 1) confusion[1][1]++;
				else confusion[0][1]++;
			}
		}
	
		std::cout << "Confusion Matrix:" << std:: endl;
		
		printf("\t%10d%8d\n", 1, 0);
		printf("\t%1d%9d%8d\n", 1, confusion[0][0], confusion[0][1]);
		printf("\t%1d%9d%8d\n", 0, confusion[1][0], confusion[1][1]);
		
		int correctClass = confusion[0][0] + confusion[1][1];
		int incorrectClass = confusion[0][1] + confusion[1][0];

		printf( "\nTest Results:\n"
		"\tCorrect classifications: %d (%g%%)\n"
		"\tIncorrect classifications: %d (%g%%)\n",
		correctClass, (double) correctClass*100/testingRows,
		incorrectClass, (double) incorrectClass*100/testingRows);
		
	}

	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	printf("\tTotal Time: %f sec.\n\n", seconds);
	return 0;
}
