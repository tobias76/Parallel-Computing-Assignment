//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) 
{
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
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

__kernel void maxValue(__global float* toSort, __local float* output, __global int* index)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int N = get_global_size(0);

	float lowestValue = 3.0;

	output[id] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if(!(localID % (i * 2)) && ((localID + i) < N))
		{ 
			if (toSort[i] > lowestValue)
			{
				lowestValue = toSort[i];
				*index = toSort[i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	//copy the cache to output array
	toSort[id] = output[id];
}

/*
These work on the smaller dataset.
*/

__kernel void averageValueGlobal(__global float* toSort, __global float* index)
{ 
	int id = get_global_id(0);
	int localID = get_local_id(0);
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
	int localID = get_local_id(0);
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