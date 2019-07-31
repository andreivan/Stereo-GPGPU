/*
Depth calculation from Stereo LR images using Adaptive Weighted Sum algorithm
*/

#define at2(arr, x, y, width) arr[x * width + y]
#define at3(arr, x, y, z, depth, width) arr[x * depth * width + y * depth + z]

// PADDING Y
int index_y(int y, int height)
{
	if (0 <= y && y < height)
		return y;
	else if (y < 0)
		return 0;
	else
		return height - 1;
}
// PADDING X
int index_x(int x, int width)
{
	if (0 <= x && x < width)
		return x;
	else if (x < 0)
		return 0;
	else
		return width - 1;
}


__kernel void ASW_Disparity(__global uchar * left, __global uchar * right,
	__global float * datacost, int width, int height, int dispRange, int mSize)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	int k = get_global_id(2);

	// Changeable const parameter
	const float sigmaColor = 10.0f;
	const float sigmaSpatial = mSize*1.5f;
	const float truncation = 25.0f;
	float spatialDiff, colorDiff, weightL, weightR;
	if (j - k >= 0)
	{
		float diff = 0;
		float temp_diff = 0;
		float weight = 0;
		for (int m = -mSize; m <= mSize; m++)
		{
			for (int n = -mSize; n <= mSize; n++)
			{
				int x1 = j + n;
				int x2 = j + n - k;
				int y = i + m;
				//Pad the index
				y = index_y(y, height);
				x1 = index_x(x1, width);
				x2 = index_x(x2, width);

				spatialDiff = hypot((float)m, (float)n);

				colorDiff =
					sqrt((float)(at2(left, y, x1, width) - at2(left, i, j, width))
						* (float)(at2(left, y, x1, width) - at2(left, i, j, width)));
				weightL = exp(-1.0f * ((colorDiff / sigmaColor)+ (spatialDiff / sigmaSpatial)));

				colorDiff =
					sqrt((float)(at2(right, y, x2, width) - at2(right, i, j - k, width))
						* (float)(at2(right, y, x2, width) - at2(right, i, j - k, width)));
				weightR = exp(-1.0f * ((colorDiff / sigmaColor) + (spatialDiff / sigmaSpatial)));

				temp_diff = fabs((float)(at2(left, y, x1, width) - at2(right, y, x2, width)));
				if (temp_diff > truncation)
					temp_diff = truncation;
				//diff += fabs((float)(at2(left, y, x1, width) - at2(right, y, x2, width)))
				//	* weightL * weightR;
				diff += temp_diff * weightL * weightR / 255.0f;
				weight += weightL * weightR;
			}
		}
		//at3(datacost, i, j, k, dispRange, width) = diff / weight;
		datacost[dispRange * (width * i + j) + k] = diff / weight;
	}
	else {
		//at3(datacost, i, j, k, dispRange, width) = FLT_MAX;
		datacost[dispRange * (width * i + j) + k] = 1000.0f;
	}
}