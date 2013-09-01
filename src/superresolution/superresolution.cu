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
)
{
	// ### Implement me ###

	/*	for(int x=0;x<nx;x++){
		int x1 = x+1; if(x1 >= nx) x1 = nx-1;
		for(int y=0;y<ny;y++){
			 int y1 = y+1; if(y1 >= ny) y1 = ny-1;
			const int p = y*nx+x;
			float dx = (xi1[p] + tau_d * factor_update * (uor[y*nx+x1] - uor[p])) /huber_denom;
			float dy = (xi2[p] + tau_d * factor_update * (uor[y1*nx+x] - uor[p])) /huber_denom;
			float denom = sqrtf(dx*dx + dy*dy)/factor_clipping;
			if(denom < 1.0f) denom = 1.0f;
			xi1[p] = dx / denom;
			xi2[p] = dy / denom;
		}
	}*/
}

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
)
{
	// ### Implement me ###

	/*for(int p=0;p<nx*ny;p++){
		dual[p] = (dual[p] + tau_d*factor_update* (primal[p] - constant[p]))/huber_denom;
		if(dual[p] < -factor_clipping) dual[p] = -factor_clipping;
		if(dual[p] > factor_clipping)  dual[p] = factor_clipping;
	}*/
}

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
)
{
	// ### Implement me ###
	/*for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			const int p = y*nx+x;
			float u_old = u[p];
			float u_new = u[p] + tau_p *
					(factor_tv_update *(xi1[p] - (x==0 ? 0.0f : xi1[p-1]) + xi2[p] - (y==0 ? 0.0f : xi2[p-nx]))
						 - factor_degrade_update * degraded[p]);
			u[p] = u_new;
			uor[p] = overrelaxation*u_new + (1.0f-overrelaxation) * u_old;
		}
	}*/
}

void computeSuperresolutionUngerGPU
(
		float *xi1_g, // TODO should this be transferred to memory? Not sure
		float *xi2_g, // TODO should this be transferred to memory? Not sure
		float *temp1_g, // TODO should this be transferred to memory? I think not.
		float *temp2_g, // TODO should this be transferred to memory? I think not.
		float *temp3_g, // TODO should this be transferred to memory? I think not.
		float *temp4_g, // TODO should this be transferred to memory? I think not.
		float *uor_g, // TODO should this be transferred to memory? I think not.
		float *u_g, //TODO quite sure this value is wrong from the function call and when returning it is the only value sent to host
		std::vector<float*> &q_g, // Already on memory
		std::vector<float*> &images_g, // Already on memory
		std::list<FlowGPU> &flowsGPU, // Already on memory
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
	// TODO Check each call to a function to see if it is a kernel or a function, if it is a kernel we must add <<<dimGrid,dimBlock>>> between function name and parameters

	//Obtain grid size and block size
	dim3 dimGrid((int)ceil((float)nx/SR_BW), (int)ceil((float)ny/SR_BH));
	dim3 dimBlock(SR_BW,SR_BH);


	//int nc = 1; // We only use one channel since images are grayscale

	fprintf(stderr,"\nComputing 1N Superresolution from %i Images on GPU",(int)images_g.size());

	/*float *u_g = (float*)result->data;*/ //TODO This should probably come from the caller function but I'm pretty sure it is messed up
	//	float *uor = u_g; // TODO This was commented before, probably should delete it.

	setKernel<<<dimGrid, dimBlock>>>(xi1_g, nx, ny, pitchf1, 0.0f );
	setKernel<<<dimGrid, dimBlock>>>(xi2_g, nx, ny, pitchf1, 0.0f );
	setKernel<<<dimGrid, dimBlock>>>(u_g, nx, ny, pitchf1, 64.0f );
	setKernel<<<dimGrid, dimBlock>>>(uor_g, nx, ny, pitchf1, 64.0f );

	for(unsigned int k=0;k<q_g.size();k++){
		setKernel<<<dimGrid, dimBlock>>>(q_g[k], nx_orig, ny_orig, pitchf1_orig, 0.0f );
	}

	float factorquad = factor_rescale_x*factor_rescale_y*factor_rescale_x*factor_rescale_y;
	float factor_degrade_update = pow(factorquad,CLIPPING_TRADEOFF_DEGRADE_1N);
	float factor_degrade_clipping = factorquad/factor_degrade_update;
	float huber_denom_degrade = 1.0f + huber_epsilon*tau_d/factor_degrade_clipping;


	float factor_tv_update = pow(factor_tv,CLIPPING_TRADEOFF_TV);
	float factor_tv_clipping = factor_tv/factor_tv_update;
	float huber_denom_tv = 1.0f + huber_epsilon*tau_d/factor_tv;

	// oi is the number of iterations
	for(int i=0;i<oi;i++){
		fprintf(stderr," %i",i);

		//DUAL TV
		dualTVHuberGPU<<<dimGrid, dimBlock>>>(uor_g,xi1_g,xi2_g,nx,ny,pitchf1,factor_tv_update,factor_tv_clipping,huber_denom_tv,tau_d);

		//DUAl DATA
		unsigned int k=0;
		std::vector<float*>::iterator image_g=images_g.begin();
		std::list<FlowGPU>::iterator flow_g = flowsGPU.begin();
		while(image_g != images_g.end() && flow_g != flowsGPU.end() && k < q_g.size()){
			float *f = *image_g;
			backwardRegistrationBilinearValueTex(uor_g,flow_g->u_g,flow_g->v_g,temp1_g,0.0f,nx,ny,pitchf1,pitchf1,1.0f,1.0f);
			if(blur > 0.0f){
				gaussBlurSeparateMirrorGpu(temp1_g,temp2_g,nx,ny,pitchf1,blur,blur,(int)(3.0f*blur),temp4_g,0);
			}
			else{
				float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
			}
			if(factor_rescale_x > 1.0f || factor_rescale_y > 1.0f){
				resampleAreaParallelSeparate(temp2_g,temp1_g,nx,ny,pitchf1,nx_orig,ny_orig,pitchf1_orig,temp4_g);
			}
			else{
				float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
			}
			dualL1DifferenceGPU<<<dimGrid, dimBlock>>>(temp1_g,f,q_g[k],nx_orig,ny_orig, pitchf1_orig,factor_degrade_update,factor_degrade_clipping,huber_denom_degrade,tau_d);
			k++;
			flow_g++;
			image_g++;
		}

		//PROX
		setKernel<<<dimGrid, dimBlock>>>(temp3_g, nx, ny, pitchf1, 0.0f );
		k=0;
		image_g = images_g.begin();
		flow_g = flowsGPU.begin();
		while(image_g != images_g.end() && flow_g != flowsGPU.end() && k < q_g.size()){
			if(factor_rescale_x > 1.0f || factor_rescale_y > 1.0f){
				resampleAreaParallelSeparateAdjoined(q_g[k],temp1_g,nx_orig,ny_orig,pitchf1_orig,nx,ny,pitchf1,temp4_g);
			}
			else{
				setKernel<<<dimGrid, dimBlock>>>(temp1_g, nx_orig, ny_orig, pitchf1_orig, 0.0f );
				addKernel<<<dimGrid, dimBlock>>>(q_g[k],temp1_g,nx_orig,ny_orig,pitchf1_orig);
			}
		}
		if(blur > 0.0f){
			gaussBlurSeparateMirrorGpu(temp1_g,temp2_g,nx,ny,pitchf1,blur,blur,(int)(3.0f*blur),temp4_g,0);
		}
		else{
			float *temp = temp1_g; temp1_g = temp2_g; temp2_g = temp;
		}
		forewardRegistrationBilinearAtomic(flow_g->u_g,flow_g->v_g,temp2_g,temp1_g,nx,ny,pitchf1);
		addKernel<<<dimGrid, dimBlock>>>(temp1_g,temp3_g,nx,ny,pitchf1); //for(int x=0;x<nx*ny;x++) temp3_g[x] += temp1_g[x];
		k++;
		flow_g++;
		image_g++;
	}
	primal1NGPU<<<dimGrid, dimBlock>>>(xi1_g,xi2_g,temp3_g,u_g,uor_g,nx,ny,pitchf1,factor_tv_update,factor_degrade_update,tau_p,overrelaxation);
}






