/*
These are not used but I tried to implement a few techniques such as strides before I fixed the 
kernels.
*/
__kernel void reduceMinValue(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int N = get_global_size(0);

	output[localID] = toSort[id];

}
__kernel void minValue(__global float* toSort, __global float* output, __global int* index)
{ 
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int N = get_global_size(0);

	float lowestValue = 0.0;

	output[localID] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++) 
	{
		if(toSort[i] < lowestValue)
		{ 
			lowestValue = toSort[i];
			*index = toSort[i];
		}		
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	//copy the cache to output array
	toSort[id] = output[id];
}

__kernel void maxValue(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int N = get_local_size(0);

	float largestValue = 0.0;

	output[localID] = toSort[id];

	//wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if((localID % (i * 2)) && ((localID + i) < N))
		{ 
			if (!(localID % (i * 2)) && ((localID + i) < N))
			{ 
				// If the current value is greater than our biggest value
				if (toSort[i] > largestValue)
				{
					largestValue = toSort[i];
					*index = toSort[i];
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//copy the cache to output array
	toSort[id] = output[id];
}

/*
These work on the large unit.
*/

// This calculates the histogram.
__kernel void histogram(__global const int* input, __global int* output, __global int* minimum)
{
	int id = get_global_id(0);
	int bin_index = input[id];

	// Use atomic incrementers to add the bin index and the minimum hist value together and add them to the output
	atomic_inc(&output[bin_index + minimum[0]]);
}

/* 
Using strides this adds all of the values together ready for dividing them on the CPU. And uses local memory.
*/
__kernel void add(__global int* input, __global  int* output, __local int* scratch)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int size = get_local_size(0);

	scratch[localID] = input[id];

	// Wait till all local threads are complete
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2)
	{
		if (!(localID % (i * 2)) && ((localID + i) < size))
		{
			scratch[localID] += scratch[localID + i];
		}
		// Wait till all local threads are complete
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// If not local ID add the current scratch to the output
	if (!localID)
	{
		atomic_add(&output[0], scratch[localID]);
	}
}

// This checks the current value in the vector to the lowest value and replaces it if it is lower.
__kernel void minValueLargeGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);

	float lowestValue = 0.0;

	// Set the corresponding output value to the current input value
	output[id] = toSort[id];

	// Wait till all global threads are complete
	barrier(CLK_GLOBAL_MEM_FENCE);

	// If the current output is smaller than the index
	if (output[id] < *index)
	{
		// Set the index to the output
		*index = output[id];
		// Set the current location as the lowest value
		lowestValue = id;
		// Wait till all global threads are complete
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// This checks the current value in the vector to the lowest value and replaces it if it is higher.
__kernel void maxValueLargeGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);

	float largestValue = 0.0;

	// Set the corresponding output value to the current input value
	output[id] = toSort[id];

	// Wait till all local threads are complete
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (output[id] > *index)
	{
		// Set the index to the output
		*index = output[id];
		// Set the current location as the highest value
		largestValue = id;
		// Wait till all global threads are complete
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// Using strides this adds all of the values together ready for dividing them on the CPU.
__kernel void averageValueLargeGlobal(__global float* toSort, __global float* index)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	float averageValue = 0.0;

	// Wait till all global threads are complete
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Add the current value with the average value.
	averageValue += toSort[id];

	// Wait till all global threads are complete
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	//*index = averageValue / id;
}

/*
These work on the smaller dataset.
*/

__kernel void averageValueGlobal(__global float* toSort, __global float* index)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	float averageValue = 0.0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++)
	{
		averageValue += toSort[i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	*index = averageValue / N;
}

__kernel void minValueGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	float lowestValue = 0.0;

	output[id] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++)
	{
		if (toSort[i] < lowestValue)
		{
			lowestValue = toSort[i];
			*index = toSort[i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	//copy the cache to output array
	toSort[id] = output[id];
}

__kernel void maxValueGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int N = get_global_size(0);

	float lowestValue = 0.0;

	output[id] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i++)
	{
		if (toSort[i] > lowestValue)
		{
			lowestValue = toSort[i];
			*index = toSort[i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	//copy the cache to output array
	toSort[id] = output[id];
}