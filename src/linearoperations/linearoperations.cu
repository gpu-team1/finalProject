/****************************************************************************\
 *      --- Practical Course: GPU Programming in Computer Vision ---
 *
 * time:    winter term 2012/13 / March 11-18, 2013
 *
 * project: superresolution
 * file:    linearoperations.cu
 *
 *
 * implement all functions with ### implement me ### in the function body
\****************************************************************************/

/*
 * linearoperations.cu
 *
 *  Created on: Aug 3, 2012
 *      Author: steinbrf
 */


#include <auxiliary/cuda_basic.cuh>
#include <linearoperations/linearoperations.cuh>

cudaChannelFormatDesc linearoperation_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;


#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   21    // maximum allowed kernel radius + 1
__constant__ float constKernel[MAXKERNELSIZE];


void setTexturesLinearOperations(int mode){
	tex_linearoperation.addressMode[0] = cudaAddressModeClamp;
	tex_linearoperation.addressMode[1] = cudaAddressModeClamp;
	if(mode == 0)tex_linearoperation.filterMode = cudaFilterModePoint;
	else tex_linearoperation.filterMode = cudaFilterModeLinear;
	tex_linearoperation.normalized = false;
}


#define LO_TEXTURE_OFFSET 0.5f
#define LO_RS_AREA_OFFSET 0.0f

#ifdef DGT400
#define LO_BW 32
#define LO_BH 16
#else
#define LO_BW 16
#define LO_BH 16
#endif


#ifndef RESAMPLE_EPSILON
#define RESAMPLE_EPSILON 0.005f
#endif

#ifndef atomicAdd
__device__ float atomicAdd(float* address, double val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val + __int_as_float(assumed)));
	}	while (assumed != old);
	return __int_as_float(old);
}

#endif

void gpu_bindConstantLinearOperationMemory(const float *kernel, int size)
{
	cutilSafeCall( cudaMemcpyToSymbol(constKernel, kernel, size*sizeof(float)) );
}



void gpu_bindLinearOperationTextureMemory(const float *in_g, int iWidth, int iHeight, size_t iPitchBytes)
{
	cutilSafeCall(cudaBindTexture2D(0, &tex_linearoperation, in_g, &linearoperation_float_tex, iWidth, iHeight, iPitchBytes) );
}


void gpu_unbindLinearOperationTextureMemory()
{
	cutilSafeCall( cudaUnbindTexture(tex_linearoperation) );
}

__global__ void gpu_backwardRegistrationBilinearValueTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		float value,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	// ### Implement me ###
	// Get current pixel
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;

	const int idx_in = x + y * pitchf1_in;
	const int idx_out = x + y * pitchf1_out;

	float hx_1 = 1.0f/hx;
	float hy_1 = 1.0f/hy;

	if(x < nx && y < ny){
		float ii_fp = x+(flow1_g[idx_in]*hx_1);
		float jj_fp = y+(flow2_g[idx_in]*hy_1);

		if((ii_fp < 0.0f) || (jj_fp < 0.0f)
				|| (ii_fp > (float)(nx-1)) || (jj_fp > (float)(ny-1))){
			out_g[idx_out] = value;
		}
		else if(!isfinite(ii_fp) || !isfinite(jj_fp)){
			//fprintf(stderr,"!"); //TODO throws error compiling
			out_g[idx_out] = value;
		}
		else{
			int xx = (int)floor(ii_fp);
			int yy = (int)floor(jj_fp);

			int xx1 = xx == nx-1 ? xx : xx+1;
			int yy1 = yy == ny-1 ? yy : yy+1;

			float xx_rest = ii_fp - (float)xx;
			float yy_rest = jj_fp - (float)yy;

			float value_xx_yy = tex2D(tex_linearoperation, xx + LO_TEXTURE_OFFSET, yy + LO_TEXTURE_OFFSET);
			float value_xx1_yy = tex2D(tex_linearoperation, xx1 + LO_TEXTURE_OFFSET, yy + LO_TEXTURE_OFFSET);
			float value_xx_yy1 = tex2D(tex_linearoperation, xx + LO_TEXTURE_OFFSET, yy1 + LO_TEXTURE_OFFSET);
			float value_xx1_yy1 = tex2D(tex_linearoperation, xx1 + LO_TEXTURE_OFFSET, yy1 + LO_TEXTURE_OFFSET);

			out_g[y*nx+x] = (1.0f-xx_rest)*(1.0f-yy_rest) * value_xx_yy
					+ xx_rest*(1.0f-yy_rest) * value_xx1_yy
					+ (1.0f-xx_rest)*yy_rest * value_xx_yy1
					+ xx_rest * yy_rest * value_xx1_yy1;

		}
	}
}

/* TODO RC3108
 * When we call this function, we must initialize the texture to the value we want, but this should be done where the call to the function is,
 * or not sure if we can actually do this in this function, only from the first thread or something like that.
 * https://devtalk.nvidia.com/default/topic/512016/file-scope-of-texture-and-surface-references/
 * setTexturesLinearOperations(0);
 * gpu_bindLinearOperationTextureMemory(in_g, iWidth, iHeight, iPitchBytes); //change iWidth, iHeight and iPitchbytes to match
 * <<<call to the kernel>>>
 * gpu_unbindLinearOperationTextureMemory();
 */
/*
 * Host function
 */
void backwardRegistrationBilinearValueTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		float value,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	/* TODO RC0109 This should be done in the file superresolution.cu, but is here as reference
	 *
	 *
	size_t pitchBytesF1, pitchF1;
	int nc = 1; // We only use one channel since images are grayscale

	// Allocation of GPU Memory (THIS CAN BE CALLED IN THE CONSTRUCTOR)
	cuda_malloc2D((void**)&(_u_overrelaxed_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
	cuda_malloc2D((void**)&(flow1_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
	cuda_malloc2D((void**)&(flow2_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
	cuda_malloc2D((void**)&(_help1_g), nx, ny, nc, sizeof(float), &pitchBytesF1);

	// Compute the pitch
	pitchF1 = pitchBytesF1/sizeof(float);
	// Copy input arrays from host to device
	cuda_copy_h2d_2D(_u_overrelaxed, _u_overrelaxed_g, nx, ny, nc, sizeof(float), pitchF1);
	cuda_copy_h2d_2D(flow_g->u1, flow1_g, nx, ny, nc, sizeof(float), pitchF1);
	cuda_copy_h2d_2D(flow->u2, flow2_g, nx, ny, nc, sizeof(float), pitchF1);

	// TODO Call the function from somewhere within the code
	backwardRegistrationBilinearValueTex(_u_overrelaxed_g,flow1_g,flow2_g,_help1_g,0.0f,_nx,_ny,pitchF1,pitchF1,1.0f,1.0f);
	// Synchronize threads
	cutilSafeCall( cudaThreadSynchronize() );

	// Copy data from device to host
	cuda_copy_d2h_2D(_help1_g,_help1, nx, ny, nc, sizeof(float), pitchF1);

	// Free memory reserved in device, this could be done in the destructor of the main CUDA function
	  if (_u_overrelaxed_g) cutilSafeCall( cudaFree(_u_overrelaxed_g) );
  	  if (flow1_g) cutilSafeCall( cudaFree(flow1_g) );
  	  if (flow2_g) cutilSafeCall( cudaFree(flow2_g) );
  	  if (_help1_g) cutilSafeCall( cudaFree(_help1_g) );

	 */

	dim3 dimGrid((int)ceil((float)nx/LO_BW), (int)ceil((float)ny/LO_BH));
	dim3 dimBlock(LO_BW,LO_BH);

	setTexturesLinearOperations(0);

	gpu_bindLinearOperationTextureMemory(in_g, nx, ny, pitchf1_in*sizeof(float) ); //change iWidth, iHeight and iPitchbytes to match
	gpu_backwardRegistrationBilinearValueTex<<<dimGrid, dimBlock>>>(in_g, flow1_g, flow2_g, out_g, value, nx, ny, pitchf1_in, pitchf1_out, hx, hy);
	gpu_unbindLinearOperationTextureMemory();
}


// ### Implement me ###
// This function doesn't need initialization or texture so it can be defined as a kernel and called by the main CUDA function directly
__global__ void backwardRegistrationBilinearFunctionGlobal
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	/* TODO RC0109 This should be done in the file flowlib_cpu_sor.cu, but is here as reference
	 *
	 *
		size_t pitchBytesF1, pitchF1;
		int nc = 1; // We only use one channel since images are grayscale

		// Allocation of GPU Memory (THIS CAN BE CALLED IN THE CONSTRUCTOR)
		cuda_malloc2D((void**)&(_I1pyramid_level_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
		cuda_malloc2D((void**)&(_I2pyramid_level_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
		cuda_malloc2D((void**)&(_u1_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
		cuda_malloc2D((void**)&(_u2_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
		cuda_malloc2D((void**)&(_I2warp_g), nx, ny, nc, sizeof(float), &pitchBytesF1);

		// Compute the pitch
		pitchF1 = pitchBytesF1/sizeof(float);
		// Copy input arrays from host to device
		cuda_copy_h2d_2D(_I1pyramid->level[rec_depth], _I1pyramid_level_g, nx, ny, nc, sizeof(float), pitchF1);
		cuda_copy_h2d_2D(_I2pyramid->level[rec_depth], _I2pyramid_level_g, nx, ny, nc, sizeof(float), pitchF1);
		cuda_copy_h2d_2D(_u1, _u1_g, nx, ny, nc, sizeof(float), pitchF1);
		cuda_copy_h2d_2D(_u2, _u2_g, nx, ny, nc, sizeof(float), pitchF1);

		// TODO Call the function from somewhere within the code
		backwardRegistrationBilinearFunctionGlobal<<<dimGrid, dimBlock>>>(_I2pyramid_level_g,_u1_g,_u2_g_I2warp,_I1pyramid_level_g,nx_fine,ny_fine,pitchF1,pitchF1,hx_fine,hy_fine);
		OR
		backwardRegistrationBilinearFunctionTex(_I2pyramid_level_g,_u1_g,_u2_g_I2warp,_I1pyramid_level_g,nx_fine,ny_fine,pitchF1,pitchF1,hx_fine,hy_fine);
		// Synchronize threads
		cutilSafeCall( cudaThreadSynchronize() );

		// Copy data from device to host
		cuda_copy_d2h_2D(_I2warp_g,_I2warp, nx, ny, nc, sizeof(float), pitchF1);

		// Free memory reserved in device, this could be done in the destructor of the main CUDA function
		  if (_I1pyramid_level_g) cutilSafeCall( cudaFree(_I1pyramid_level_g) );
		  if (_I2pyramid_level_g) cutilSafeCall( cudaFree(_I2pyramid_level_g) );
	  	  if (_u1_g) cutilSafeCall( cudaFree(_u1_g) );
	  	  if (_u2_g) cutilSafeCall( cudaFree(_u2_g) );
	  	  if (_I2warp_g) cutilSafeCall( cudaFree(_I2warp_g) );

	 */

	// get current pixel coordinates
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;

	const int idx_in = x + y * pitchf1_in;
	const int idx_out = x + y * pitchf1_out;

	if(x < nx && y < ny){
		const float xx = (float)x+flow1_g[idx_in]/hx;
		const float yy = (float)y+flow2_g[idx_in]/hy;

		int xxFloor = (int)floor(xx);
		int yyFloor = (int)floor(yy);

		int xxCeil = xxFloor == nx-1 ? xxFloor : xxFloor+1;
		int yyCeil = yyFloor == ny-1 ? yyFloor : yyFloor+1;

		float xxRest = xx - (float)xxFloor;
		float yyRest = yy - (float)yyFloor;

		out_g[idx_out] =
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
				? constant_g[idx_in] :
						(1.0f-xxRest)*(1.0f-yyRest) * in_g[yyFloor*pitchf1_in+xxFloor]
						                                   + xxRest*(1.0f-yyRest) * in_g[yyFloor*pitchf1_in+xxCeil]
						                                                                 + (1.0f-xxRest)*yyRest * in_g[yyCeil*pitchf1_in+xxFloor]
						                                                                                               + xxRest*yyRest * in_g[yyCeil*pitchf1_in+xxCeil];
	}
}



// ### Implement me, if you want ###
__global__ void gpu_backwardRegistrationBilinearFunctionTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	// get current pixel coordinates
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;

	const int idx_in = x + y * pitchf1_in;
	const int idx_out = x + y * pitchf1_out;

	if(x < nx && y < ny){
		const float xx = (float)x+flow1_g[idx_in]/hx;
		const float yy = (float)y+flow2_g[idx_in]/hy;

		int xxFloor = (int)floor(xx);
		int yyFloor = (int)floor(yy);

		int xxCeil = xxFloor == nx-1 ? xxFloor : xxFloor+1;
		int yyCeil = yyFloor == ny-1 ? yyFloor : yyFloor+1;

		float xxRest = xx - (float)xxFloor;
		float yyRest = yy - (float)yyFloor;

		float value_xxF_yyF = tex2D(tex_linearoperation, xxFloor + LO_TEXTURE_OFFSET, yyFloor + LO_TEXTURE_OFFSET);
		float value_xxC_yyF = tex2D(tex_linearoperation, xxCeil + LO_TEXTURE_OFFSET, yyFloor + LO_TEXTURE_OFFSET);
		float value_xxF_yyC = tex2D(tex_linearoperation, xxFloor + LO_TEXTURE_OFFSET, yyCeil + LO_TEXTURE_OFFSET);
		float value_xxC_yyC = tex2D(tex_linearoperation, xxCeil + LO_TEXTURE_OFFSET, yyCeil + LO_TEXTURE_OFFSET);
		out_g[idx_out] =
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
				? constant_g[idx_in] :
						(1.0f-xxRest)*(1.0f-yyRest) * value_xxF_yyF
						+ xxRest*(1.0f-yyRest) * value_xxC_yyF
						+ (1.0f-xxRest)*yyRest * value_xxF_yyC
						+ xxRest*yyRest * value_xxF_yyF;
	}
}

void backwardRegistrationBilinearFunctionTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	/* TODO RC0109 This should be done in the file flowlib_cpu_sor.cu, but is here as reference
			 *
			 *
			size_t pitchBytesF1, pitchF1;
			int nc = 1; // We only use one channel since images are grayscale

			// Allocation of GPU Memory (THIS CAN BE CALLED IN THE CONSTRUCTOR)
			cuda_malloc2D((void**)&(_I1pyramid_level_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
			cuda_malloc2D((void**)&(_I2pyramid_level_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
			cuda_malloc2D((void**)&(_u1_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
			cuda_malloc2D((void**)&(_u2_g), nx, ny, nc, sizeof(float), &pitchBytesF1);
			cuda_malloc2D((void**)&(_I2warp_g), nx, ny, nc, sizeof(float), &pitchBytesF1);

			// Compute the pitch
			pitchF1 = pitchBytesF1/sizeof(float);
			// Copy input arrays from host to device
			cuda_copy_h2d_2D(_I1pyramid->level[rec_depth], _I1pyramid_level_g, nx, ny, nc, sizeof(float), pitchF1);
			cuda_copy_h2d_2D(_I2pyramid->level[rec_depth], _I2pyramid_level_g, nx, ny, nc, sizeof(float), pitchF1);
			cuda_copy_h2d_2D(_u1, _u1_g, nx, ny, nc, sizeof(float), pitchF1);
			cuda_copy_h2d_2D(_u2, _u2_g, nx, ny, nc, sizeof(float), pitchF1);

			// TODO Call the function from somewhere within the code
			backwardRegistrationBilinearFunctionGlobal<<<dimGrid, dimBlock>>>(_I2pyramid_level_g,_u1_g,_u2_g_I2warp,_I1pyramid_level_g,nx_fine,ny_fine,pitchF1,pitchF1,hx_fine,hy_fine);
			OR
			backwardRegistrationBilinearFunctionTex(_I2pyramid_level_g,_u1_g,_u2_g_I2warp,_I1pyramid_level_g,nx_fine,ny_fine,pitchF1,pitchF1,hx_fine,hy_fine);
			// Synchronize threads
			cutilSafeCall( cudaThreadSynchronize() );

			// Copy data from device to host
			cuda_copy_d2h_2D(_I2warp_g,_I2warp, nx, ny, nc, sizeof(float), pitchF1);

			// Free memory reserved in device, this could be done in the destructor of the main CUDA function
			  if (_I1pyramid_level_g) cutilSafeCall( cudaFree(_I1pyramid_level_g) );
			  if (_I2pyramid_level_g) cutilSafeCall( cudaFree(_I2pyramid_level_g) );
		  	  if (_u1_g) cutilSafeCall( cudaFree(_u1_g) );
		  	  if (_u2_g) cutilSafeCall( cudaFree(_u2_g) );
		  	  if (_I2warp_g) cutilSafeCall( cudaFree(_I2warp_g) );

			 */

	dim3 dimGrid((int)ceil((float)nx/LO_BW), (int)ceil((float)ny/LO_BH));
	dim3 dimBlock(LO_BW,LO_BH);

	setTexturesLinearOperations(0);
	gpu_bindLinearOperationTextureMemory(in_g, nx, ny, pitchf1_in*sizeof(float) );
	gpu_backwardRegistrationBilinearFunctionTex<<<dimGrid, dimBlock>>>(in_g, flow1_g, flow2_g, out_g, constant_g, nx, ny, pitchf1_in, pitchf1_out, hx, hy);
	gpu_unbindLinearOperationTextureMemory();
}

// ### Implement me ###
__global__ void gpu_forewardRegistrationBilinearAtomic
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
		float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
)
{
	// get current pixel coordinates
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;

	if( x < nx && y < ny ){

		const int idx = x + y * pitchf1;
		const float xx = (float)x + flow1_g[idx];
		const float yy = (float)y + flow2_g[idx];
		if(xx >= 0.0f && xx <= (float)(nx-2) && yy >= 0.0f && yy <= (float)(ny-2))
		{
			float xxf = floor(xx);
			float yyf = floor(yy);
			const int xxi = (int)xxf;
			const int yyi = (int)yyf;
			xxf = xx - xxf;
			yyf = yy - yyf;
			float value = in_g[idx];
			double value1 = value * (1.0f-xxf)*(1.0f-yyf);
			double value2 = value * xxf*(1.0f-yyf);
			double value3 = value * (1.0f-xxf)*yyf;
			double value4 = value * xxf*yyf;

			atomicAdd(&out_g[yyi*nx+xxi], value1);
			atomicAdd(&out_g[yyi*nx+xxi+1], value2);
			atomicAdd(&out_g[(yyi+1)*nx+xxi], value3);
			atomicAdd(&out_g[(yyi+1)*nx+xxi+1], value4);
		}
	}
}

/* The out_g array that is an input to the following function must be initialized to 0, which can be done by calling the following
 * kernel before:
 * setKernel( float *field_g, nx,	ny, pitchf1, 0.0f )
 */
void forewardRegistrationBilinearAtomic
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
		float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
)
{
	dim3 dimGrid((int)ceil((float)nx/LO_BW), (int)ceil((float)ny/LO_BH));
	dim3 dimBlock(LO_BW,LO_BH);

	setKernel<<<dimGrid, dimBlock>>>(out_g, nx, ny, pitchf1, 0.0f );
	gpu_forewardRegistrationBilinearAtomic<<<dimGrid, dimBlock>>>(flow1_g, flow2_g, in_g, out_g, nx, ny, pitchf1);
}

/*
float *makeGaussianKernel(int kRadiusX, int kRadiusY, float sigmaX, float sigmaY)
{
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;
  const int kernelSize = kWidth*kHeight;
  float *kernel = new float[kernelSize];

  // ### build a normalized gaussian kernel ###

  // kernel at (x,y): kernel[(x+kRadiusX) + (y+kRadiusY)*kWidth]
  // x = -kRadiusX .. kRadiusX
  // y = -kRadiusY .. kRadiusY

  float total=0.0;

  for (int x=-kRadiusX; x <= kRadiusX; ++x){
    for (int y=-kRadiusY; y <= kRadiusY; ++y){
      kernel[(x+kRadiusX) + (y+kRadiusY)*kWidth] = (1/(2*M_PI*sigmaY*sigmaX))*exp(-1/2*((x*x/(sigmaX*sigmaX))+(y*y/(sigmaY*sigmaY))));
      total += kernel[(x+kRadiusX) + (y+kRadiusY)*kWidth];
    }
  }

  for (int x=-kRadiusX; x <= kRadiusX; ++x){
    for (int y=-kRadiusY; y <= kRadiusY; ++y){
      kernel[(x+kRadiusX) + (y+kRadiusY)*kWidth] = kernel[(x+kRadiusX) + (y+kRadiusY)*kWidth]/ total;
    }
  }

  return kernel;
}

void gpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  size_t iPitchBytes, kPitchBytes;
  size_t iPitch, kPitch;
    float *d_kernel;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);

  // allocate device memory
  cutilSafeCall( cudaMallocPitch( (void**)&d_kernel, &kPitchBytes, kWidth*sizeof(float), kHeight ) );
  kPitch = kPitchBytes/sizeof(float);
  //std::cout << "iPitchBytes=" << iPitchBytes << " iPitch=" << iPitch << " kPitchBytes=" << kPitchBytes << " kPitch=" << kPitch << std::endl;

  cutilSafeCall( cudaMemcpy2D(d_kernel, kPitchBytes, kernel, kWidth*sizeof(float), kWidth*sizeof(float), kHeight, cudaMemcpyHostToDevice) );


  gpu_bindConstantMemory(kernel, kWidth*kHeight);

  dim3 blockSize(BW,BH);
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );


  gpu_convolutionGrayImage_gm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
 cutilSafeCall( cudaThreadSynchronize() );


  // free memory
  cutilSafeCall( cudaFree(d_kernel) );
}
*/
__global__ void gaussBlurSeparateMirrorGpuKernel
(
		float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float sigmax,
		float sigmay,
		int   radius,
		float *temp_g,
		float *mask
)
{
	/*// get current pixel coordinates
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;

		if( x < nx && y < ny ){}

	// ### Implement me ###
	if(sigmax <= 0.0f || sigmay <= 0.0f || radius < 0){
		return;
	}

	if(radius == 0){
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)(3.0f*maxsigma);
	}

	bool selfalloctemp = temp == NULL;
	if(selfalloctemp) temp = new float[nx*ny];
	bool selfallocmask = mask == NULL;
	if(selfallocmask) mask = new float[radius+1];

	float result, sum;
	sigmax = 1.0f/(sigmax*sigmax);
	sigmay = 1.0f/(sigmay*sigmay);

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmax));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			result = mask[0]*in_g[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((x-i>=0) ? in_g[y*nx+(x-i)] : in_g[y*nx+(-1-(x-i))]) +
						((x+i<nx) ? in_g[y*nx+(x+i)] : in_g[y*nx+(nx-(x+i-(nx-1)))]));
			}
			temp[y*nx+x] = result;
		}
	}

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmay));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++)	{
			result = mask[0]*temp[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((y-i>=0) ? temp[(y-i)*nx+x] : temp[(-1-(y-i))*nx+x]) +
						((y+i<ny) ? temp[(y+i)*nx+x] : temp[(ny-(y+i-(ny-1)))*nx+x]));
			}
			out_g[y*nx+x] = result;
		}
	}

	if(selfallocmask) delete [] mask;
	if(selfalloctemp) delete [] temp;*/

}
void gaussBlurSeparateMirrorGpu
(
		float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float sigmax,
		float sigmay,
		int   radius,
		float *temp_g,
		float *mask
)
{
	/*// get current pixel coordinates
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;

		if( x < nx && y < ny ){}

	// ### Implement me ###
	if(sigmax <= 0.0f || sigmay <= 0.0f || radius < 0){
		return;
	}

	if(radius == 0){
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)(3.0f*maxsigma);
	}

	bool selfalloctemp = temp == NULL;
	if(selfalloctemp) temp = new float[nx*ny];
	bool selfallocmask = mask == NULL;
	if(selfallocmask) mask = new float[radius+1];

	float result, sum;
	sigmax = 1.0f/(sigmax*sigmax);
	sigmay = 1.0f/(sigmay*sigmay);

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmax));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			result = mask[0]*in_g[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((x-i>=0) ? in_g[y*nx+(x-i)] : in_g[y*nx+(-1-(x-i))]) +
						((x+i<nx) ? in_g[y*nx+(x+i)] : in_g[y*nx+(nx-(x+i-(nx-1)))]));
			}
			temp[y*nx+x] = result;
		}
	}

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmay));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++)	{
			result = mask[0]*temp[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((y-i>=0) ? temp[(y-i)*nx+x] : temp[(-1-(y-i))*nx+x]) +
						((y+i<ny) ? temp[(y+i)*nx+x] : temp[(ny-(y+i-(ny-1)))*nx+x]));
			}
			out_g[y*nx+x] = result;
		}
	}

	if(selfallocmask) delete [] mask;
	if(selfalloctemp) delete [] temp;*/

}
/*
void gaussBlurSeparateMirror
(
		float *in,
		float *out,
		int   nx,
		int   ny,
		float sigmax,
		float sigmay,
		int   radius,
		float *temp,
		float *mask
)
{
	if(sigmax <= 0.0f || sigmay <= 0.0f || radius < 0){
		return;
	}

	if(radius == 0){
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)(3.0f*maxsigma);
	}

	bool selfalloctemp = temp == NULL;
	if(selfalloctemp) temp = new float[nx*ny];
	bool selfallocmask = mask == NULL;
	if(selfallocmask) mask = new float[radius+1];

	float result, sum;
	sigmax = 1.0f/(sigmax*sigmax);
	sigmay = 1.0f/(sigmay*sigmay);

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmax));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			result = mask[0]*in[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((x-i>=0) ? in[y*nx+(x-i)] : in[y*nx+(-1-(x-i))]) +
						((x+i<nx) ? in[y*nx+(x+i)] : in[y*nx+(nx-(x+i-(nx-1)))]));
			}
			temp[y*nx+x] = result;
		}
	}

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmay));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++)	{
			result = mask[0]*temp[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((y-i>=0) ? temp[(y-i)*nx+x] : temp[(-1-(y-i))*nx+x]) +
						((y+i<ny) ? temp[(y+i)*nx+x] : temp[(ny-(y+i-(ny-1)))*nx+x]));
			}
			out[y*nx+x] = result;
		}
	}

	if(selfallocmask) delete [] mask;
	if(selfalloctemp) delete [] temp;
}

*/


void resampleAreaParallelSeparate
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g,
		float scalefactor
)
{
	// ### Implement me ###
}
/*
void resampleAreaParallelizableSeparate
(
		const float *in,
		float *out,
		int   nx_in,
		int   ny_in,
		int   nx_out,
		int   ny_out,
		float *help,
		float scalefactor
)
{
	bool selfalloc = help == 0;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		help = new float[std::max(nx_in,nx_out)*std::max(ny_in,ny_out)];
	}
	resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,
			(float)(nx_in)/(float)(nx_out),nx_in,(float)(nx_out)/(float)(nx_in));
	resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,
			(float)(ny_in)/(float)(ny_out),scalefactor*(float)(ny_out)/(float)(ny_in));
	if(selfalloc){
		delete [] help;
	}
}

void resampleAreaParallelizableSeparate_x
(
		const float *in,
		float *out,
		int   nx,
		int   ny,
		float hx,
		int   nx_orig,
		float factor = 0.0f
)
{

	if(factor == 0.0f) factor = 1.0f/hx;

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			int p = y*nx+x;

			float px = (float)x * hx;
			float left = ceil(px) - px;
			if(left > hx) left = hx;
			float midx = hx - left;
			float right = midx - floorf(midx);
			midx = midx - right;

			out[p] = 0.0f;

			if(left > 0.0f){
				out[p] += in[y*nx_orig+(int)(floor(px))]*left*factor;
				px+= 1.0f;
			}
			while(midx > 0.0f){
				out[p] += in[y*nx_orig+(int)(floor(px))]*factor;
				px += 1.0f;
				midx -= 1.0f;
			}
			if(right > RESAMPLE_EPSILON)	{
				out[p] += in[y*nx_orig+(int)(floor(px))]*right*factor;
			}
		}
	}
}


void resampleAreaParallelizableSeparate_y
(
		const float *in,
		float *out,
		int   nx,
		int   ny,
		float hy,
		float factor = 0.0f
)
{
	if(factor == 0.0f) factor = 1.0f/hy;

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			int p = y*nx+x;
			float py = (float)y * hy;
			float top = ceil(py) - py;
			if(top > hy) top = hy;
			float midy = hy - top;
			float bottom = midy - floorf(midy);
			midy = midy - bottom;

			out[p] = 0.0f;

			if(top > 0.0f){
				out[p] += in[(int)(floor(py))*nx+x]*top*factor;
				py += 1.0f;
			}
			while(midy > 0.0f){
				out[p] += in[(int)(floor(py))*nx+x]*factor;
				py += 1.0f;
				midy -= 1.0f;
			}
			if(bottom > RESAMPLE_EPSILON){
				out[p] += in[(int)(floor(py))*nx+x]*bottom*factor;
			}
		}
	}
}
*/
void resampleAreaParallelSeparateAdjoined
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g,
		float scalefactor
)
{
	// ### Implement me ###
}

/*
void resampleAreaParallelizableSeparateAdjoined
(
		const float *in,
		float *out,
		int   nx_in,
		int   ny_in,
		int   nx_out,
		int   ny_out,
		float *help,
		float scalefactor
)
{
	bool selfalloc = help == 0;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		help = new float[std::max(nx_in,nx_out)*std::max(ny_in,ny_out)];
	}
	resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,(float)(nx_in)/(float)(nx_out),nx_in,1.0f);
	resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,(float)(ny_in)/(float)(ny_out),scalefactor);
	if(selfalloc){
		delete [] help;
	}
}
*/
// ### Implement me ###
__global__ void addKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	// Get current pixel
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	// Check if the pixel is inside the boundaries
	if(x < nx && y < ny)
	{
		// Compute linearized index
		int idx = x + y * pitchf1;
		// Perform vector addition
		accumulator_g[idx] += increment_g[idx];
	}
}

// ### Implement me ###
__global__ void subKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	// Get current pixel
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	// Check if the pixel is inside the boundaries
	if(x < nx && y < ny)
	{
		// Compute linearized index
		int idx = x + y * pitchf1;
		// Perform vector subtraction
		accumulator_g[idx] -= increment_g[idx];
	}
}

// ### Implement me ###
__global__ void setKernel
(
		float *field_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float value
)
{
	// Get current pixel
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	// Check if the pixel is inside the boundaries
	if(x < nx && y < ny)
	{
		// Compute linearized index
		int idx = x + y * pitchf1;
		// Perform vector subtraction
		field_g[idx] = value;
	}
}

