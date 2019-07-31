#define at2(arr, x, y, width) arr[x * width + y]
#define at3(arr, x, y, z, depth, width) arr[x * depth * width + y * depth + z]

__kernel void winner_takes_all(__global float * datacost, __global uchar * dispFinal,
	int width, int height, int dispRange, int dispScale)
{
	//int j = get_global_id(0);
	//int i = get_global_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int j = tx + get_group_id(0) * get_local_size(0);
	int i = ty + get_group_id(1) * get_local_size(1);


	float minCost = FLT_MAX;
	float cost = 0.0f;
	uchar disparity = 0;
	{
	#pragma unroll
		for (int d = 0; d < dispRange; ++d)
		{
			//cost = at3(datacost, i, j, d, dispRange, width);
			cost = datacost[dispRange * (width * i + j) + d];
			if (minCost > cost)
			{
				minCost = cost;
				disparity = d;
			}
		}
		at2(dispFinal, i, j, width) = disparity * (uchar)dispScale;
	}
}