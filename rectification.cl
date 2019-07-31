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

// Stereo camera parameter HD
float constant F_RECTI = 264.29444970357366f * 2.0f;
float constant CX_RECTI = 321.6046943664551f * 2.0f;
float constant CY_RECTI = 238.9259433746338f * 2.0f;

float constant FX = 413.074435f * 2.0f; 
float constant FY = 413.53896f * 2.0f;  
float constant CX = 299.831845f * 2.0f; 
float constant CY = 234.01305f * 2.0f;  

float constant FX2 = 821.59003f;
float constant FY2 = 823.07792f; 
float constant CX2 = 718.06715f;
float constant CY2 = 484.56333f;
////////////////////////////////////////////////////////////////////

//Stereo camera no distortion parameter
//float constant F_RECTI_VGA = 532.66379f;
//float constant CX_RECTI_VGA = 308.27207f;
//float constant CY_RECTI_VGA = 211.22144f;
//float constant FX_VGA = 531.48547f;
//float constant FY_VGA = 535.38115f;
//float constant CX_VGA = 317.36588f;
//float constant CY_VGA = 231.51589f;
//float constant K1 = 0.10394f;
//float constant K2 = -0.31141f;
//float constant P1 = -0.01591f;
//float constant P2 = 0.00107f;
//
//float constant FX2_VGA = 528.39448f;
//float constant FY2_VGA = 532.66379f;
//float constant CX2_VGA = 325.58373f;
//float constant CY2_VGA = 221.11119f;
//float constant K11 = 0.10519f;
//float constant K22 = -0.20104f;
//float constant P11 = -0.01859f;
//float constant P22 = -0.00516f;

//Stereo camera with distortion parameter
float constant F_RECTI_VGA = 264.29444970357366f;
float constant CX_RECTI_VGA = 321.6046943664551f;
float constant CY_RECTI_VGA = 238.9259433746338f;
float constant FX_VGA = 413.074435f; 
float constant FY_VGA = 413.53896f ;  
float constant CX_VGA = 299.831845f; 
float constant CY_VGA = 234.01305f ; 
float constant K1 = -0.3857f;
float constant K2 = 0.18076f;
float constant P1 = 0.00004f;
float constant P2 = -0.00188f;

float constant FX2_VGA = 410.795015f; 
float constant FY2_VGA = 411.53896f ; 
float constant CX2_VGA = 359.033575f; 
float constant CY2_VGA = 242.281665f; 
float constant K11 = -0.36101f;
float constant K22 = 0.12757f;
float constant P11 = -0.00036f;
float constant P22 = -0.00112f;

//Stereo camera no distortion parameter2
//float constant F_RECTI_VGA = 264.29444970357366f;
//float constant CX_RECTI_VGA = 321.6046943664551f;
//float constant CY_RECTI_VGA = 238.9259433746338f;
//float constant FX_VGA = 420.34273f;
//float constant FY_VGA = 421.97661f;
//float constant CX_VGA = 331.07862f;
//float constant CY_VGA = 230.79201f;
//float constant K1 = -0.38584f;
//float constant K2 = 0.20258f;
//float constant P1 = 0.00178f;
//float constant P2 = -0.00064f;
//
//float constant FX2_VGA = 421.17524f;
//float constant FY2_VGA = 422.85973f;
//float constant CX2_VGA = 325.68780f;
//float constant CY2_VGA = 233.11305f;
//float constant K11 = -0.38543f;
//float constant K22 = 0.19216f;
//float constant P11 = 0.00123f;
//float constant P22 = 0.00038f;


__kernel void rectify(__global uchar*  in,
	__global uchar*  Left,
	__global uchar*  Right,
	int    width,
	int    height)
{
	int xt = get_global_id(0);
	int yt = get_global_id(1);

	float coeffx = 0.0f;
	float xd = (xt - CX_RECTI) / F_RECTI;
	float yd = (yt - CY_RECTI) / F_RECTI;
	float x = xd;
	float y = yd;
	int i = 0;
	int j = 0;
	float a = 0;
	float b = 0;
	float r_sqr = x * x + y * y;
	float result_ = 0.0f;

	if (xt < width / 2)
	{
		coeffx = 1 + K1 * r_sqr + K2 * r_sqr * r_sqr;

		x = x * coeffx + 2 * P1 * x * y + P2 * (r_sqr + (2 * x * x));
		y = y * coeffx + P1 * (r_sqr + (2 * y * y) + 2 * P2 * x * y);

		x = FX * x + CX;
		y = FY * y + CY;

		if (x >= 1 && x <= width / 2 && y >= 1 && y < height)
		{
			i = (int)x;
			a = x - i;
			j = (int)y;
			b = y - j;

			result_ = (1.0f - a) * (1.0f - b) * (float)(in[(j - 1)*width + (i - 1)])
				+ a * (1.0f - b) * (float)(in[(j - 1)*width + i])
				+ a * b * (float)(in[j*width + i])
				+ (1.0f - a) * b * (float)(in[j*width + (i - 1)]);
			Left[(yt - 1) * width / 2 + (xt - 1)] = convert_uchar_sat(result_);
		}
		x = xd;
		y = yd;

		coeffx = 1 + K11 * r_sqr + K22 * r_sqr * r_sqr;

		x = x * coeffx + 2 * P11 * x * y + P22 * (r_sqr + (2 * x * x));
		y = y * coeffx + P11 * (r_sqr + (2 * y * y) + 2 * P22 * x * y);

		x = FX2 * x + CX2;
		y = FY2 * y + CY2;

		if (x >= 0 && x < width / 2 && y >= 0 && y < height)
		{
			x = x + width / 2;
			i = (int)(x);
			a = x - i;
			j = (int)(y);
			b = y - j;

			result_ = (1.0f - a) * (1.0f - b) * (float)(in[(j - 1)*width + (i - 1)])
				+ a * (1.0f - b) * (float)(in[(j - 1)*width + i])
				+ a * b * (float)(in[j*width + i])
				+ (1.0f - a) * b * (float)(in[j*width + (i - 1)]);
			Right[(yt - 1) * width / 2 + (xt - 1)] = convert_uchar_sat(result_);
		}
	}
}

__kernel void rectify_VGA(__global uchar*  in,
	__global uchar*  Left,
	__global uchar*  Right,
	int    width,
	int    height)
{
	int xt = get_global_id(0);
	int yt = get_global_id(1);

	float coeffx = 0.0f;
	float xd = (xt - CX_RECTI_VGA) / F_RECTI_VGA;
	float yd = (yt - CY_RECTI_VGA) / F_RECTI_VGA;
	float x = xd;
	float y = yd;
	int i = 0;
	int j = 0;
	float a = 0;
	float b = 0;
	float r_sqr = x * x + y * y;
	float result_ = 0.0f;

	if (xt < width / 2)
	{
		coeffx = 1 + K1 * r_sqr + K2 * r_sqr * r_sqr;

		x = x * coeffx + 2 * P1 * x * y + P2 * (r_sqr + (2 * x * x));
		y = y * coeffx + P1 * (r_sqr + (2 * y * y) + 2 * P2 * x * y);

		x = FX_VGA * x + CX_VGA;
		y = FY_VGA * y + CY_VGA;

		if (x >= 1 && x <= width / 2 && y >= 1 && y < height)
		{
			i = (int)x;
			a = x - i;
			j = (int)y;
			b = y - j;

			result_ = (1.0f - a) * (1.0f - b) * (float)(in[(j - 1)*width + (i - 1)])
				+ a * (1.0f - b) * (float)(in[(j - 1)*width + i])
				+ a * b * (float)(in[j*width + i])
				+ (1.0f - a) * b * (float)(in[j*width + (i - 1)]);
			Left[(yt - 1) * width / 2 + (xt - 1)] = convert_uchar_sat(result_);
		}
		x = xd;
		y = yd;

		coeffx = 1 + K11 * r_sqr + K22 * r_sqr * r_sqr;

		x = x * coeffx + 2 * P11 * x * y + P22 * (r_sqr + (2 * x * x));
		y = y * coeffx + P11 * (r_sqr + (2 * y * y) + 2 * P22 * x * y);

		x = FX2_VGA * x + CX2_VGA;
		y = FY2_VGA * y + CY2_VGA;

		if (x >= 0 && x < width / 2 && y >= 0 && y < height)
		{
			x = x + width / 2;
			i = (int)(x);
			a = x - i;
			j = (int)(y);
			b = y - j;

			result_ = (1.0f - a) * (1.0f - b) * (float)(in[(j - 1)*width + (i - 1)])
				+ a * (1.0f - b) * (float)(in[(j - 1)*width + i])
				+ a * b * (float)(in[j*width + i])
				+ (1.0f - a) * b * (float)(in[j*width + (i - 1)]);
			Right[(yt - 1) * width / 2 + (xt - 1)] = convert_uchar_sat(result_);
		}
	}
}


__kernel void rectify_back(__global uchar*  input,
	__global uchar*  Left,
	int    width,
	int    height)
{
	int xt = get_global_id(0);
	int yt = get_global_id(1);

	float coeffx = 0.0f;

	float xd = (xt - CX_VGA) / FX_VGA;
	float yd = (yt - CY_VGA) / FY_VGA;

	float x = xd;
	float y = yd;
	int i = 0;
	int j = 0;
	float a = 0;
	float b = 0;
	float r_sqr = x * x + y * y;
	float result_ = 0.0f;
	
	coeffx = (1 - K1 * r_sqr) + ((3 * K1 * K1 - 2 * K1*K2) * r_sqr * r_sqr);
	x = x * coeffx;
	y = y * coeffx;

	x = F_RECTI_VGA * x + CX_RECTI_VGA;
	y = F_RECTI_VGA * y + CY_RECTI_VGA;
	
	if (x >= 1 && x <= width && y >= 1 && y < height)
	{
		i = (int)x;
		a = x - i;
		j = (int)y;
		b = y - j;

		result_ = (1.0f - a) * (1.0f - b) * (float)(input[(j - 1)*width + (i - 1)])
			+ a * (1.0f - b) * (float)(input[(j - 1)*width + i])
			+ a * b * (float)(input[j*width + i])
			+ (1.0f - a) * b * (float)(input[j*width + (i - 1)]);
		Left[(yt - 1) * width + (xt - 1)] = convert_uchar_sat(result_);
	}


	//Left[yt * width + xt] = input[yt*width + xt];

}
