#include <cv.h>
#include <ml.h>
#include <stdio.h>

using namespace cv;

#define NUMBER_OF_ATTRIBUTES 10
#define NUMBER_OF_CLASSES 2

int main( int argc, char** argv )
{
	FileStorage fs;
		
	if(!fs.open("data.xml", FileStorage::READ)) {
		printf("Error: Could not locate 'data.xml'\nExiting...\n");
		return -1;
	};
	
	//printf ("Using OpenCV version %s\n", CV_VERSION);
	
	clock_t t1,t2,t3;
	t1=clock();
	
	Mat trainingData;
	Mat trainingClasses;

	Mat testingData;
	Mat testingClasses;
	
	fs["trainingData"] >> trainingData;
	fs["trainingClasses"] >> trainingClasses;
	fs["testingData"] >> testingData;
	fs["testingClasses"] >> testingClasses;
	
	int trainingRows = trainingData.rows;
	int testingRows = testingData.rows;

	double result;
		
	CvSVM* svm = new CvSVM;

	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);

	svm->train( trainingData, trainingClasses, Mat(), Mat(), params);

	t2=clock();

	Mat testSample;
	
	int confusion[2][2] = {0,0,0,0};

	for (int tsample = 0; tsample < testingRows; tsample++)
	{

		testSample = testingData.row(tsample);

		result = svm->predict(testSample);

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
	std::cout << "\nSVM" << std::endl;
	std::cout << "\nConfusion Matrix:" << std:: endl;
	
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

	t3=clock();
	float diff1 ((float)t2-(float)t1);
	float diff2 ((float)t3-(float)t2);
	float seconds1 = diff1 / CLOCKS_PER_SEC;
	float seconds2 = diff2 / CLOCKS_PER_SEC;
	printf("\tModeling: %f sec.\n", seconds1);
	printf("\tTesting: %f sec.\n\n", seconds2);

	svm->save("trainedSVM.xml", "svm");

	return 0;
}
