/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    superresolution.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * superresolution.cuh
 *
 *  Created on: May 16, 2012
 *      Author: steinbrf
 */
#ifndef SUPERRESOLUTION_CUH
#define SUPERRESOLUTION_CUH
#include <vector>
#include <list>

/*!
 * Class for Storing a 2D displacement field on the GPU
 */
class FlowGPU
{
public:
	//! Frees the GPU Memory
	void clear();
	//! Horizontal Component
	float *u_g;
	//! Vertical Component
	float *v_g;
	//! Width
	int   nx;
	//! Height
	int   ny;
};

__global__ void dualTVHuberGPU
(
		const float *uor,
		float       *xi1,
		float       *xi2,
		int         nx,
		int         ny,
		int			pitchf1,
		float       factor_update,
		float       factor_clipping,
		float       huber_denom,
		float       tau_d
);

__global__ void dualL1DifferenceGPU
(
		const float *primal,
		const float *constant,
		float       *dual,
		int         nx,
		int         ny,
		int   		pitchf1,
		float       factor_update,
		float       factor_clipping,
		float       huber_denom,
		float       tau_d
);

__global__ void primal1NGPU
(
		const float *xi1,
		const float *xi2,
		const float *degraded,
		float       *u,
		float       *uor,
		int         nx,
		int         ny,
		int			pitchf1,
		float       factor_tv_update,
		float       factor_degrade_update,
		float       tau_p,
		float       overrelaxation
);

void computeSuperresolutionUngerGPU
(
		float *xi1_g,
		float *xi2_g,
		float *temp1_g,
		float *temp2_g,
		float *temp3_g,
		float *temp4_g,
		float *uor_g,
		float *u_g,
		std::vector<float*> &q_g,
		std::vector<float*> &images_g,
		std::list<FlowGPU> &flowsGPU,
		int   &nx,
		int   &ny,
		int   &pitchf1,
		int   &nx_orig,
		int   &ny_orig,
		int   &pitchf1_orig,
		int   &oi,
		float &tau_p,
		float &tau_d,
		float &factor_tv,
		float &huber_epsilon,
		float &factor_rescale_x,
		float &factor_rescale_y,
		float &blur,
		float &overrelaxation,
		int   debug
);



#endif
