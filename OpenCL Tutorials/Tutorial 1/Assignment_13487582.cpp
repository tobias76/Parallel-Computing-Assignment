#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <ctime>
#include <CL/cl.hpp>

#include "Utils.h"
#include "Variables.h"

using namespace std;

void print_help() 
{ 
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

// This pads the vector making it divisible by the amount of workgroups
void padVector()
{
	// This calculates how much padding the array needs.
	int paddingSize = temperatureVector.size() % local_size;

	// This array is added to the end of the temperature array as padding.
	vector<int> arrayExtension(local_size - paddingSize, 0);

	if (paddingSize)
	{
		// Add the vector extension to the end of the temperature vector
		temperatureVector.insert(temperatureVector.end(), arrayExtension.begin(), arrayExtension.end());
	}
}

void fileRead()
{
	cout << "DEBUG: Loading in file: " << endl;

	// This is where the file is.
	ifstream File("../../Data/temp_lincolnshire.txt");

	try
	{
		// Whilst there is text in the file.
		while (getline(File, currentLine))
		{
			// Create a string stream of the current line in the file.
			stringstream stream(currentLine);

			// Split the values at the space
			stream >> currentLocation >> currentYear >> currentMonth >> currentDay >> currentTime >> currentTemperature;

			// Allocate the values to the correct variables
			locationVector.push_back(currentLocation);
			monthVector.push_back(currentMonth);
			yearVector.push_back(currentYear);
			dayVector.push_back(currentDay);
			timeVector.push_back(currentTime);
			// I am using two different arrays to allow for both float and int manipulation
			temperatureVector.push_back(currentTemperature);
			averageTemperature.push_back(currentTemperature);
		}
	}
	catch (const std::exception&)
	{
		cout << "Error reading file in" << endl;
	}
}

int main(int argc, char **argv) 
{
	/*
		Please find any NON-OPEN CL variables and their definitions in the aptly named Variables.h header
	*/ 

	// Initilise two clocks, one for the file read in and one for the full execution time.
	startTime = clock();
	fullLength = clock();

	typedef int mytype;

	// Read the text file into the vectors
	fileRead();

	// Pad the vectors to the correct sizes
	padVector();

	// Calculate how long the loop took to complete
	duration = (clock() - startTime) / (double)CLOCKS_PER_SEC;

	cout << "PERF: File Read In: " << duration << "secs" << endl;
	cout << "DEBUG: Data Size: " << locationVector.size() << endl;
	cout << "DEBUG: Recommend Work Group Size " << CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE << endl;

	//Number of elements in the vector
	vector_elements = temperatureVector.size();
	//size in bytes
	vectorByteSize = temperatureVector.size() * sizeof(mytype);
	// Size of workgroups
	numberOfGroups = vector_elements / local_size;

	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try 
	{
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "DEBUG: Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);
		
		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels13487582.cl");

		cl::Program program(context, sources);

		try
		{
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) 
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}		

		//5.1 Create a vector of the correct size to store the output of the GPU calcs.
		outputVector.resize(vector_elements);

		////device - buffers
		cout << "DEBUG: " << "Creating buffers" << endl;
		cl::Buffer temperatureBuffer(context, CL_MEM_READ_ONLY, vectorByteSize);
		cl::Buffer averageTempBuff(context, CL_MEM_READ_ONLY, vectorByteSize);
		cl::Buffer output(context, CL_MEM_READ_WRITE, vectorByteSize);
		cl::Buffer maxBuffer(context, CL_MEM_READ_WRITE, intSize);
		cl::Buffer minimumBuffer(context, CL_MEM_READ_WRITE, intSize);
		cl::Buffer averageBuffer(context, CL_MEM_READ_WRITE, intSize);

		// Create two buffers of the temperature buffer, one as ints and one as floats.
		queue.enqueueWriteBuffer(temperatureBuffer, CL_TRUE, 0, vectorByteSize, &temperatureVector[0], 0, NULL);
		queue.enqueueWriteBuffer(averageTempBuff, CL_TRUE, 0, vectorByteSize, &averageTemperature[0], 0, NULL);
 
		// This is used to make sure I am running the correct vector.
		cout << "DEBUG: " << "Vector size: " << temperatureVector.size() << endl;

		// These kernels are ran further in the program.
		cl::Kernel maxValueKernel = cl::Kernel(program, "maxValueLargeGlobal");
		cl::Kernel minValueKernel = cl::Kernel(program, "minValueLargeGlobal");
		cl::Kernel averageKernel = cl::Kernel(program, "add");
		cl::Kernel histogramKernel = cl::Kernel(program, "histogram");

		/* Give arguments to the kernel, these are usually some varient of:
		   0 - Input
		   1 - Output Vector
		   2 - Singular output.
		*/
		maxValueKernel.setArg(0, temperatureBuffer);
		maxValueKernel.setArg(1, output);
		maxValueKernel.setArg(2, maxBuffer);
		
		minValueKernel.setArg(0, temperatureBuffer);
		minValueKernel.setArg(1, output);
		minValueKernel.setArg(2, minimumBuffer);

		averageKernel.setArg(0, averageTempBuff);
		averageKernel.setArg(1, averageBuffer);
		// Creates a LocalSpaceArg of the size of two 
		averageKernel.setArg(2, cl::Local(local_size * intSize));

		try
		{
			cout << "DEBUG: " << "Adding kernels to the command queue " << endl;
			// Here we add the kernel to the queue 
			cout << "DEBUG: " << "Running the maxValueKernel " << endl;
			queue.enqueueNDRangeKernel(maxValueKernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
			cout << "DEBUG: " << "Running the minValueKernel " << endl;
			queue.enqueueNDRangeKernel(minValueKernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
			cout << "DEBUG: " << "Running the averageKernel " << endl;
			queue.enqueueNDRangeKernel(averageKernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
		}
		catch (const std::exception&)
		{
			cout << "Error: Executing Kernels" << endl;
		}

		try
		{
			cout << "DEBUG: " << "Reading results from GPU into variables " << endl;
			////5.3 Copy the result from the GPU to host
			queue.enqueueReadBuffer(output, CL_TRUE, 0, vectorByteSize, &outputVector[0]);
			queue.enqueueReadBuffer(averageBuffer, CL_TRUE, 0, intSize, &average);
			queue.enqueueReadBuffer(maxBuffer, CL_TRUE, 0, floatSize, &largeIndex);
			queue.enqueueReadBuffer(minimumBuffer, CL_TRUE, 0, floatSize, &index);
		}
		catch (const std::exception&)
		{
			cout << "Error: Reading from buffers" << endl;
		}

		// This converts the integer sum of values to a float and divides them by the size of the temperature vector
		averageTemp = (float)average / temperatureVector.size();

		// This creates a histogram vector.
		histogram.resize(largeIndex - index + 1);
		
		// Allocates a memory space the size of the vector 
		histogramSize = histogram.size() * intSize;

		// This inverts the index and allocates it to the minimum of the histogram
		histogramMinimum = index *- 1;

		// Create further buffers on the device for the Histogram
		
		cout << "DEBUG: " << "Creating Histogram Buffers" << endl;
		cl::Buffer bufferTempHist(context, CL_MEM_READ_WRITE, vectorByteSize);
		cl::Buffer bufferTemp(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer bufferHistMin(context, CL_MEM_READ_WRITE, histogramSize * intSize);

		// Create and fill the histogram buffers
		queue.enqueueWriteBuffer(bufferTempHist, CL_TRUE, 0, vectorByteSize, &averageTemperature[0]);
		
		queue.enqueueFillBuffer(bufferTemp, CL_TRUE, 0, 0);

		queue.enqueueWriteBuffer(bufferHistMin, CL_TRUE, 0, histogram.size(), &histogramMinimum);

		histogramKernel.setArg(0, bufferTempHist);
		histogramKernel.setArg(1, bufferTemp);
		histogramKernel.setArg(2, bufferHistMin);


		try
		{
			cout << "Queueing Histogram Kernel" << endl;
			queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
		}
		catch (const std::exception&)
		{
			cout << "Error: Queuing Histogram Kernel" << endl;
		}

		try
		{
			cout << "Reading Histogram Buffer into Variable" << endl;
			queue.enqueueReadBuffer(bufferTemp, CL_TRUE, 0, histogramSize, &histogram[0]);
		}
		catch (const std::exception&)
		{
			cout << "Error: Reading Histogram Buffer" << endl;
		}

		std::cout << "OUTPUT: Average Value = " << averageTemp << endl;
		std::cout << "OUTPUT: Max Value = " << largeIndex << endl;
		std::cout << "OUTPUT: Min Value = " << index << endl;
		std::cout << "OUTPUT: Histogram Bin Values = " << histogram << endl;

	}
	catch (cl::Error Error) 
	{
		cerr << "ERROR: " << Error.what() << ", " << getErrorString(Error.err()) << endl;
	}

	// Calculate how long the whole program took to complete by subtracting the clock from the starttime and dividing it by the clocks per second and converting it to a double.
	fullDuration = (clock() - startTime) / (double)CLOCKS_PER_SEC;
	cout << "PERF: Program Executed In: " << fullDuration << "secs" << endl;

	return 0;
}