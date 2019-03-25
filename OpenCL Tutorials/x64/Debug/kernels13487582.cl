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

	float lowestValue = 0.0;

	output[localID] = toSort[id];

	//wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if((localID % (i * 2)) && ((localID + i) < N))
		{ 
			if (!(localID % (i * 2)) && ((localID + i) < N))
			{ 
				if (toSort[i] > lowestValue)
				{
					lowestValue = toSort[i];
					*index = toSort[i];
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//copy the cache to output array
	toSort[id] = output[id];
}

__kernel void histogram(__global const int* input, __global int* output, __global int* minimum)
{ 
	int id = get_global_id(0);
	int bin_index = input[id];

	atomic_inc(&output[bin_index + minimum[0]])
}

/*
These work on the large unit
*/

__kernel void add(__global int* input, __global  int* output, __local int* scratch)
{
	int id = get_global_id(0);
	int localID = get_local_id(0);
	int size = get_local_size(0);

	scratch[localID] = input[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2)
	{
		if (!(localID % (i * 2)) && ((localID + i) < size))
		{
			scratch[localID] += scratch[localID + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!localID)
	{
		atomic_add(&output[0], scratch[localID]);
	}
}

__kernel void minValueLargeGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);

	float lowestValue = 0.0;

	output[id] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (output[id] < *index)
	{
		*index = output[id];
		lowestValue = id;
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void maxValueLargeGlobal(__global float* toSort, __global float* output, __global int* index)
{
	int id = get_global_id(0);

	float lowestValue = 0.0;

	output[id] = toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (output[id] > *index)
	{
		*index = output[id];
		lowestValue = id;
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void averageValueLargeGlobal(__global float* toSort, __global float* index)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	float averageValue = 0.0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	averageValue += toSort[id];

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	*index = averageValue / id;
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