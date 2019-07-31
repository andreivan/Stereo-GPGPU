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

#define at2(arr, x, y, width) arr[x * width + y]

//Gausian 5x5 kernel
float constant gaussian_kernel[25] = { 0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f,
0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
0.000922f, 0.107391f, 0.466066f, 0.107391f, 0.000922f,
0.000212f, 0.024745f, 0.107391f, 0.024745f, 0.000212f,
0.000002f, 0.000212f, 0.000922f, 0.000212f, 0.000002f };

//Sobel 5x5 kernel X & Y direction
float constant sobel_kernelY[25] = { -0.41666f,  -0.03333f,  0.0f,  0.03333f,  0.41666f,
-0.06666f,  -0.08333f,  0.0f,  0.08333f,  0.06666f,
-0.08333f,  -0.16666f,  0.0f,  0.16666f,  0.08333f,
-0.06666f,  -0.08333f,  0.0f,  0.08333f,  0.06666f,
-0.41666f,  -0.03333f,  0.0f,  0.03333f,  0.41666f };

float constant sobel_kernelX[25] = { -0.41666f, -0.06666f,  -0.08333f, -0.06666f,  -0.41666f,
-0.03333f, -0.08333f,  -0.16666f, -0.08333f,  -0.03333f,
0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
0.03333f, 0.08333f,  0.16666f, 0.08333f,  0.03333f,
0.41666f, 0.06666f,  0.08333f, 0.06666f,  0.41666f };


//Gaussian filter on image
__kernel void gaussian_filter(__global uchar* inputL,
	__global uchar* inputR,
	__global uchar*  gaussianL,
	__global uchar*  gaussianR)
{

	int xt = get_global_id(0);
	int yt = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int i = 0, j = 0;

	if (xt > 2 && xt < width - 2 && yt > 2 && yt < height - 2)
	{
		int index = 0;
		float resultL_x = 0.0f;
		float resultR_x = 0.0f;
		for (int m = -2; m <= 2; m++)
		{
			for (int n = -2; n <= 2; n++)
			{
				i = xt + m;
				j = yt + n;
				
				resultL_x += gaussian_kernel[index] * convert_float(at2(inputL, j, i, width));
				resultR_x += gaussian_kernel[index] * convert_float(at2(inputR, j, i, width));
				index++;

			}
		}
		at2(gaussianL, yt, xt, width) = convert_uchar_sat(resultL_x);
		at2(gaussianR, yt, xt, width) = convert_uchar_sat(resultR_x);
	}
}

//Sobel filter on image
__kernel void sobel_filter(__global uchar* inputL,
	__global uchar* inputR,
	__global uchar*  sobelL,
	__global uchar*  sobelR)
{

	int xt = get_global_id(0);
	int yt = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int i = 0, j = 0;

	if (xt > 2 && xt < width - 2 && yt > 2 && yt < height - 2)
	{
		int index = 0;
		float resultL_x = 0.0f;
		float resultR_x = 0.0f;
		float resultL_y = 0.0f;
		float resultR_y = 0.0f;
		for (int m = -2; m <= 2; m++)
		{
			for (int n = -2; n <= 2; n++)
			{
				i = xt + m;
				j = yt + n;
				resultL_x += sobel_kernelX[index] * convert_float(at2(inputL, j, i, width));
				resultR_x += sobel_kernelX[index] * convert_float(at2(inputR, j, i, width));
				resultL_y += sobel_kernelY[index] * convert_float(at2(inputL, j, i, width));
				resultR_y += sobel_kernelY[index] * convert_float(at2(inputR, j, i, width));
				index++;

			}
		}
		resultL_x = hypot(resultL_x, resultL_y);
		at2(sobelL, yt, xt, width) = convert_uchar_sat(resultL_x);

		resultR_x = hypot(resultR_x, resultR_y);
		at2(sobelR, yt, xt, width) = convert_uchar_sat(resultR_x);
	}
}
