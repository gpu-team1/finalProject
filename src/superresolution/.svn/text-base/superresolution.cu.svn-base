/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    superresolution.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/

/*
 * superresolution.cu
 *
 *  Created on: May 16, 2012
 *      Author: steinbrf
 */
#include "superresolution.cuh"
#include <stdio.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include <auxiliary/cuda_basic.cuh>
#include <vector>
#include <list>

//#include <linearoperations.cuh>
#include <linearoperations/linearoperations.cuh>

#include "superresolution_definitions.h"

#include <auxiliary/debug.hpp>


#ifdef DGT400
#define SR_BW 32
#define SR_BH 16
#else
#define SR_BW 16
#define SR_BH 16
#endif

#include <linearoperations/linearoperations.h>


extern __shared__ float smem[];




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
)
{
	//### Implement me###
}





