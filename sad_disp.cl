/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
/*
Depth calculation from Stereo LR images using Sum of Absolute Distance algorithm
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

__kernel void SAD_Disparity(__global uchar * left, __global uchar * right,
	__global float * datacost, int width, int height, int dispRange, int mSize)
{
	//int j = get_global_id(0);
	//int i = get_global_id(1);
	//int k = get_global_id(2);

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tz = get_local_id(2);
	int j = tx + get_group_id(0) * get_local_size(0);
	int i = ty + get_group_id(1) * get_local_size(1);
	int k = tz + get_group_id(2) * get_local_size(2);

	int x, y;
    uint diff = 0;
    uint L, R;
	if (j - k >= 0)
	{
		diff = 0;
		for (int m = -mSize; m <= mSize; m++)
		{
			for (int n = -mSize; n <= mSize; n++)
			{
                y = i + m;
                y = index_y(y, height);

				x = j + n;
				x = index_x(x, width);
                L = at2(left, y, x, width);

                x = j + n - k;
                x = index_x(x, width);
                R = at2(right, y, x, width);

                diff += abs_diff(L , R);
			}
		}
		//at3(datacost, i, j, k, dispRange, width) = diff;
		datacost[dispRange * (width * i + j) + k] = diff / 255.0f;
	}
	else {
		//at3(datacost, i, j, k, dispRange, width) = FLT_MAX;
		datacost[dispRange * (width * i + j) + k] = 1000.0f;
	}
}
