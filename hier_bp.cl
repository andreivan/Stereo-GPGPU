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
#define HIER_ITER 4
#define DISP_RANGE 64
#define DISP_CAMERA 32
#define LOCAL_SIZE 8

//Interpolate the cost into a hier form
__kernel void initializeHierCost(__global float * datacost, __global float *hierCost,
	int width, int height, int dispRange)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	int d = get_global_id(2);
	
	if ((i * 2 + 1) >= (height*2-1) || (j * 2 + 1) >= (width*2-1))
	{
		hierCost[(dispRange * (width * i + j) + d)] = 0.0f;
	}
	else
	{
		hierCost[(dispRange * (width * i + j) + d)] =
			datacost[dispRange * ((width * 2) * (i * 2) + (j * 2)) + d] +
			datacost[dispRange * ((width * 2) * (i * 2 + 1) + (j * 2)) + d] +
			datacost[dispRange * ((width * 2) * (i * 2) + (j * 2 + 1)) + d] +
			datacost[dispRange * ((width * 2) * (i * 2 + 1) + (j * 2 + 1)) + d];

		//hierCost[(dispRange * (width * i + j) + d)] = 123.0f;
	}
}

void gpu_comp_msg(__global float *m1, __global float *m2, __global float *m3, __global float *data, __global float *dest, int MAX_DISP, float DISCONT_COST, float CONT_COST)
{
	float acc;
	float prev, cur, tmp;
	float minimum;
	int q;

	dest[0] = m1[0] + m2[0] + m3[0] + data[0];
	minimum = dest[0];

	for (q = 1; q < MAX_DISP; q++)
	{
		prev = dest[q - 1] + CONT_COST;
		cur = m1[q] + m2[q] + m3[q] + data[q];
		tmp = (prev < cur) ? prev : cur;
		dest[q] = tmp;
		minimum = (tmp < minimum) ? tmp : minimum;
	}
	minimum += DISCONT_COST;

	dest[MAX_DISP - 1] = (minimum < dest[MAX_DISP - 1]) ? minimum : dest[MAX_DISP - 1];
	acc = dest[MAX_DISP - 1];

	for (q = MAX_DISP - 2; q >= 0; q--)
	{
		prev = dest[q + 1] + CONT_COST;
		prev = (minimum < prev) ? minimum : prev;
		dest[q] = (prev < dest[q]) ? prev : dest[q];
		acc += dest[q];
	}

	acc /= (float)MAX_DISP;
	for (q = 0; q < MAX_DISP; q++)
	{
		dest[q] -= acc;
	}
}

void gpu_comp_msg_local(__local float *m1, __local float *m2, __local float *m3, float *data, float *dest, int MAX_DISP, float DISCONT_COST, float CONT_COST)
{
	float acc;
	float prev, cur, tmp;
	float minimum;
	int q;

	dest[0] = m1[0] + m2[0] + m3[0] + data[0];
	minimum = dest[0];

	for (q = 1; q < MAX_DISP; q++)
	{
		prev = dest[q - 1] + CONT_COST;
		cur = m1[q] + m2[q] + m3[q] + data[q];
		tmp = (prev < cur) ? prev : cur;
		dest[q] = tmp;
		minimum = (tmp < minimum) ? tmp : minimum;
	}
	minimum += DISCONT_COST;

	dest[MAX_DISP - 1] = (minimum < dest[MAX_DISP - 1]) ? minimum : dest[MAX_DISP - 1];
	acc = dest[MAX_DISP - 1];

	for (q = MAX_DISP - 2; q >= 0; q--)
	{
		prev = dest[q + 1] + CONT_COST;
		prev = (minimum < prev) ? minimum : prev;
		dest[q] = (prev < dest[q]) ? prev : dest[q];
		acc += dest[q];
	}

	acc /= (float)MAX_DISP;
	for (q = 0; q < MAX_DISP; q++)
	{
		dest[q] -= acc;
	}
}

//Belief message propagation on hier cost
__kernel void hierBP(
	__global float *temp_m_u, __global float *temp_m_d, __global float *temp_m_l, __global float *temp_m_r,
	__global float * datacost,
	__global float *m_u, __global float *m_d, __global float *m_l, __global float *m_r, __global float *gpu_zero,
	int width, int height, int dispRange, int dist, float DISCONT_COST, float CONT_COST)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int x = tx + get_group_id(0) * get_local_size(0);
	int y = ty + get_group_id(1) * get_local_size(1);

	__global float *temp_u;
	__global float *temp_l;
	__global float *temp_d;
	__global float *temp_r;

	temp_u = (y + 1 >= height) ? gpu_zero : &m_u[((width * dist) * (y + 1)*dist + x * dist)*dispRange];
	temp_l = (x + 1 >= width) ? gpu_zero : &m_l[((width * dist) * (y)*dist + (x + 1)*dist)*dispRange];
	temp_r = (x - 1 < 0) ? gpu_zero : &m_r[((width * dist) * (y)*dist + (x - 1)*dist)*dispRange];
	temp_d = (y - 1 < 0) ? gpu_zero : &m_d[((width * dist) * (y - 1)*dist + x * dist)*dispRange];

	//gpu_comp_msg(temp_u, temp_l, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_u[((width*scale) * (y)*dist + x * dist)*dispRange], dispRange);
	//gpu_comp_msg(temp_d, temp_l, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_d[((width*scale) * (y)*dist + x * dist)*dispRange], dispRange);
	//gpu_comp_msg(temp_u, temp_d, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_l[((width*scale) * (y)*dist + x * dist)*dispRange], dispRange);
	//gpu_comp_msg(temp_u, temp_d, temp_l, &datacost[(width * (y)+x)*dispRange], &temp_m_r[((width*scale) * (y)*dist + x * dist)*dispRange], dispRange);

	gpu_comp_msg(temp_u, temp_l, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_u[((width*dist) * (y)*dist + x * dist)*dispRange], dispRange, DISCONT_COST, CONT_COST);
	gpu_comp_msg(temp_d, temp_l, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_d[((width*dist) * (y)*dist + x * dist)*dispRange], dispRange, DISCONT_COST, CONT_COST);
	gpu_comp_msg(temp_u, temp_d, temp_r, &datacost[(width * (y)+x)*dispRange], &temp_m_r[((width*dist) * (y)*dist + x * dist)*dispRange], dispRange, DISCONT_COST, CONT_COST);
	gpu_comp_msg(temp_u, temp_d, temp_l, &datacost[(width * (y)+x)*dispRange], &temp_m_l[((width*dist) * (y)*dist + x * dist)*dispRange], dispRange, DISCONT_COST, CONT_COST);
}

__kernel void hierBP_local(
	__global float *temp_m_u, __global float *temp_m_d, __global float *temp_m_l, __global float *temp_m_r,
	__global float * datacost,
	__global float *m_u, __global float *m_d, __global float *m_l, __global float *m_r, __global float *gpu_zero,
	int width, int height, int dispRange, int dist, float DISCONT_COST, float CONT_COST)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tz = get_global_id(2);

	int x = tx + get_group_id(0) * get_local_size(0);
	int y = ty + get_group_id(1) * get_local_size(1);

	__global float *temp_u;
	__global float *temp_l;
	__global float *temp_d;
	__global float *temp_r;

	int index;
	float shr_datacost[DISP_CAMERA];	// Can't use global

	__local float shr_m_u[LOCAL_SIZE][LOCAL_SIZE][DISP_CAMERA + 1];
	__local float shr_m_l[LOCAL_SIZE][LOCAL_SIZE][DISP_CAMERA + 1];
	__local float shr_m_d[LOCAL_SIZE][LOCAL_SIZE][DISP_CAMERA + 1];
	__local float shr_m_r[LOCAL_SIZE][LOCAL_SIZE][DISP_CAMERA + 1];

	float shr_temp[DISP_CAMERA];		// Can't use global

	if ((y%dist) == 0 && y < height && (x%dist) == 0 && x < width)
	{
		index = (width*(y)+x)*DISP_CAMERA;
		for (int d = 0; d<DISP_CAMERA; d++)
		{
			shr_datacost[d] = datacost[index + d];
		}

		switch (tz)
		{
		case 0:
			// temp_u = (y + dist >= height) ? gpu_zero : &m_u[(width*(y + dist) + x)*MAX_DISP];
			temp_u = (y + 1 >= height) ? gpu_zero : &m_u[((width * dist) * (y + 1)*dist + x * dist)*DISP_CAMERA];
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				shr_m_u[ty][tx][d] = temp_u[d];
			}
			break;

		case 1:
			//temp_d = (y - dist < 0) ? gpu_zero : &m_d[(width*(y - dist) + x)*MAX_DISP];
			temp_d = (y - 1 < 0) ? gpu_zero : &m_d[((width * dist) * (y - 1)*dist + x * dist)*DISP_CAMERA];
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				shr_m_d[ty][tx][d] = temp_d[d];
			}
			break;

		case 2:
			//temp_r = (x - dist < 0) ? gpu_zero : &m_r[(width*(y)+x - dist)*MAX_DISP];
			temp_r = (x - 1 < 0) ? gpu_zero : &m_r[((width * dist) * (y)*dist + (x - 1)*dist)*DISP_CAMERA];
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				shr_m_r[ty][tx][d] = temp_r[d];
			}
			break;

		case 3:
			//temp_l = (x + dist >= width) ? gpu_zero : &m_l[(width*(y)+x + dist)*MAX_DISP];
			temp_l = (x + 1 >= width) ? gpu_zero : &m_l[((width * dist) * (y)*dist + (x + 1)*dist)*DISP_CAMERA];
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				shr_m_l[ty][tx][d] = temp_l[d];
			}
			break;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		index = ((width) * (y)+x)*DISP_CAMERA;

		switch (tz)
		{
		case 0:
			gpu_comp_msg_local(shr_m_u[ty][tx], shr_m_l[ty][tx], shr_m_r[ty][tx], shr_datacost, shr_temp, DISP_CAMERA, DISCONT_COST, CONT_COST);
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				temp_m_u[index + d] = shr_temp[d];
			}
			break;

		case 1:
			gpu_comp_msg_local(shr_m_d[ty][tx], shr_m_l[ty][tx], shr_m_r[ty][tx], shr_datacost, shr_temp, DISP_CAMERA, DISCONT_COST, CONT_COST);
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				temp_m_d[index + d] = shr_temp[d];
			}
			break;

		case 2:
			gpu_comp_msg_local(shr_m_u[ty][tx], shr_m_d[ty][tx], shr_m_r[ty][tx], shr_datacost, shr_temp, DISP_CAMERA, DISCONT_COST, CONT_COST);
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				temp_m_r[index + d] = shr_temp[d];
			}
			break;

		case 3:
			gpu_comp_msg_local(shr_m_u[ty][tx], shr_m_d[ty][tx], shr_m_l[ty][tx], shr_datacost, shr_temp, DISP_CAMERA, DISCONT_COST, CONT_COST);
			for (int d = 0; d<DISP_CAMERA; d++)
			{
				temp_m_l[index + d] = shr_temp[d];
			}
			break;
		}
	}
}

//Update cost to next layer
__kernel void updateCostLayer(__global float * m_u, __global float *m_d, __global float *m_l, __global float *m_r,
	int width, int height, int dispRange, int dist)
{
	// Use original width and height
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int x = tx + get_group_id(0) * get_local_size(0);
	int y = ty + get_group_id(1) * get_local_size(1);
	
	for (int oi = 0; oi < dist; oi++)
	{
		for (int oj = 0; oj < dist; oj++)
		{
			int k = y * dist + oi;
			int l = x * dist + oj;
			if (k <= height || l <= width)
			{
				for (int d = 0; d < dispRange; d++)
				{
					m_u[(k*width + l) * dispRange + d] = m_u[(y*dist*width + x * dist) * dispRange + d];
					m_l[(k*width + l) * dispRange + d] = m_l[(y*dist*width + x * dist) * dispRange + d];
					m_d[(k*width + l) * dispRange + d] = m_d[(y*dist*width + x * dist) * dispRange + d];
					m_r[(k*width + l) * dispRange + d] = m_r[(y*dist*width + x * dist) * dispRange + d];
				}
			}
		}
	}
}

__kernel void finalBP(__global float * m_u, __global float *m_d, __global float *m_l, __global float *m_r, 
	__global float *datacost, __global float *gpu_zero,
	int width, int height, int dispRange)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int x = tx + get_group_id(0) * get_local_size(0);
	int y = ty + get_group_id(1) * get_local_size(1);

	__global float *temp_u;
	__global float *temp_l;
	__global float *temp_d;
	__global float *temp_r;

	temp_u = (y != height - 1) ? &m_u[((y + 1)*width + x)*dispRange] : gpu_zero;
	temp_d = (y != 0) ? &m_d[((y - 1)*width + x)*dispRange] : gpu_zero;
	temp_l = (x != width - 1) ? &m_l[(y*width + x + 1)*dispRange] : gpu_zero;
	temp_r = (x != 0) ? &m_r[(y*width + x - 1)*dispRange] : gpu_zero;

	float temp;
	for (int k = 0; k < dispRange; k++)
	{
		temp = temp_u[k]
			+ temp_d[k]
			+ temp_l[k]
			+ temp_r[k]
			+ datacost[(y*width + x)*dispRange + k];

		datacost[(y*width + x)*dispRange + k] = temp;
	}
}


