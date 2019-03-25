#pragma once

#include <vector>;

using namespace std;

// Minimum value
int index = 0;
// Maximum value
int largeIndex = 0;
int histogramMinimum;

// The average temperature value
int average = 0;

size_t local_size = 2;

size_t workgroupSize = CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;

/*
	For data storage
*/

// Declare different vectors for the data
vector<string> locationVector;
vector<int> yearVector;
vector<int> monthVector;
vector<int> dayVector;
vector<int> timeVector;
vector<int> averageTemperature;
vector<float> temperatureVector;
vector<int> histogram;
vector<float> outputVector;

/*
	For reading data in
*/

// Used for taking current value in the file
string currentLocation;
int currentYear;
int currentMonth;
int currentDay;
int currentTime;
float currentTemperature;
int averageTemperatureInt;
// Current line in the file.
string currentLine;

/*
	For time checking 
*/


// These are two clocks used to calculate the time used.
clock_t startTime;
clock_t fullLength;

// These store the length of time taken to run both the read in and the full loop.
double duration = 0;
double fullDuration = 0;

/*
	For storing calculated values
*/
float averageTemp;
/*
	Variables and vectors in memory sizes
*/

// These are the size of a singular int and float, these will be used later in the program when I am needing to define the sizes of buffers.
size_t intSize = sizeof(int);
size_t floatSize = sizeof(float);

// Gets a memory space the size of the vector 
size_t histogramSize;

/*
	Vector Sizes
*/

//number of elements in the vector
size_t vector_elements;
//Size of the vector in bytes
size_t vectorByteSize;
// The number / size of our workgroups
size_t numberOfGroups;