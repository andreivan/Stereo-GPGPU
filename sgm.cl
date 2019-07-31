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

//SAD
//float constant PENALTY1 = 1.0f;		// a.k.a SMALL PENALTY
//float constant PENALTY2 = 0.1f;		// a.k.a LARGE PENALTY

//ASW
//float constant PENALTY1 = 0.05f;		// a.k.a SMALL PENALTY
//float constant PENALTY2 = 0.01f;		// a.k.a LARGE PENALTY

//#define MMAX(a,b) (((a)>(b))?(a):(b))
//#define MMIN(a,b) (((a)<(b))?(a):(b))

__kernel void evaluate_path_ocl(__global float *prior_cost, __global float *local_cost,
	float path_intensity_gradient, __global float *curr_cost, int width, int height, int disp_range, int d,
	float PENALTY1, float PENALTY2)
{
	float min_int = FLT_MAX;

	float e_smooth = FLT_MAX;
	for (int d_p = 0; d_p < disp_range; d_p++) {

		if (prior_cost[d_p]<min_int) min_int = prior_cost[d_p];

		if (d_p - d == 0) {
			// No penality
			e_smooth = fmin(e_smooth, prior_cost[d_p]);
		}
		else if (abs(d_p - d) == 1) {
			// Small penality
			e_smooth = fmin(e_smooth, prior_cost[d_p] + PENALTY1);
		}
		else {
			// Large penality
			float penalty = PENALTY2 / path_intensity_gradient;
			//penalty = fmin(penalty, PENALTY2);
			e_smooth = fmin(e_smooth, prior_cost[d_p] + fmax(PENALTY1, penalty));
		}
	}
	curr_cost[d] = local_cost[d] + e_smooth - min_int;
}

__kernel void iterate_direction_dirxpos_ocl(int dirx, __global uchar *left_image, __global float* costs,
	__global float *accumulated_costs, int width, int height, int disp_range,
	float PENALTY1, float PENALTY2)
{
	int j = get_global_id(0);	// row slice
	int d = get_global_id(1);

	if (j < height && d < disp_range) {

		for (int i = 0; i < width; i++) {
			if (i == 0)
				accumulated_costs[disp_range * (width * j + 0) + d] += costs[disp_range * (width * j + 0) + d];
			else 
			{
				evaluate_path_ocl(&accumulated_costs[disp_range * (width * j + (i - dirx)) + 0],
					&costs[disp_range * (width * j + i) + 0],
					abs(left_image[width * j + i] - left_image[width * j + (i - dirx)]),
					&accumulated_costs[disp_range * (width * j + i) + 0], width, height, disp_range, d, PENALTY1, PENALTY2);
			}
		}
	}
}

__kernel void iterate_direction_dirypos_ocl(int diry, __global uchar *left_image, __global float* costs,
	__global float *accumulated_costs, int width, int height, int disp_range,
	float PENALTY1, float PENALTY2)
{
	int i = get_global_id(0);	// col slice
	int d = get_global_id(1);

	if (i < width && d < disp_range) {

		for (int j = 0; j < height; j++) {
			if (j == 0)
				accumulated_costs[disp_range * (width * 0 + i) + d] += costs[disp_range * (width * 0 + i) + d];
			else
			{
				evaluate_path_ocl(&accumulated_costs[disp_range * (width * (j - diry) + i) + 0],
					&costs[disp_range * (width * j + i) + 0],
					abs(left_image[width*j + i] - left_image[width*(j - diry) + i]),
					&accumulated_costs[disp_range * (width * j + i) + 0], width, height, disp_range, d, PENALTY1, PENALTY2);
			}
		}
	}
}

__kernel void iterate_direction_dirxneg_ocl(int dirx, __global uchar *left_image, __global float* costs,
	__global float *accumulated_costs, int width, int height, int disp_range,
	float PENALTY1, float PENALTY2)
{
	int j = get_global_id(0);	// row slice
	int d = get_global_id(1);

	if (j < height && d < disp_range) {

		for (int i = width - 1; i >= 0; i--) {
			if (i == width - 1) 
				accumulated_costs[disp_range * (width * j + (width - 1)) + d] += costs[disp_range * (width * j + (width - 1)) + d];
			else
			{
				evaluate_path_ocl(&accumulated_costs[disp_range * (width * j + (i - dirx)) + 0],
					&costs[disp_range * (width * j + i) + 0],
					abs(left_image[width*j + i] - left_image[width*j + (i - dirx)]),
					&accumulated_costs[disp_range * (width * j + i) + 0], width, height, disp_range, d, PENALTY1, PENALTY2);
			}
		}
	}
}

__kernel void iterate_direction_diryneg_ocl(int diry, __global uchar *left_image, __global float* costs,
	__global float *accumulated_costs, int width, int height, int disp_range,
	float PENALTY1, float PENALTY2)
{
	//i  = width
	//j = height
	int i = get_global_id(0);	// col slice
	int d = get_global_id(1);

	if (i < width && d < disp_range) {
		
		for (int j = height - 1; j >= 0; j--) {
			if (j == height - 1)
				accumulated_costs[disp_range * (width * (height - 1) + i) + d] += costs[disp_range * (width * (height - 1) + i) + d];
			else
			{
				evaluate_path_ocl(&accumulated_costs[disp_range * (width * (j - diry) + i) + 0],
					&costs[disp_range * (width * j + i) + 0],
					abs(left_image[width*j + i] - left_image[width*(j - diry) + i]),
					&accumulated_costs[disp_range * (width * j + i) + 0], width, height, disp_range, d, PENALTY1, PENALTY2);
			}	
		}
	}
}

__kernel void inplace_sum_views_ocl(__global float * im1, __global float * im2, int width, int dispRange)
{
	int j = get_global_id(0);	// 0 ~ width * height * dispRange
	int i = get_global_id(1);
	int d = get_global_id(2);
	im1[dispRange * (width * i + j) + d] += im2[dispRange * (width * i + j) + d];
	//im1[j] += im2[j]
}
