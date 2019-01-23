#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592
#define ini -0.7071067811865475244008
#define b_TPBX 16
#define b_TPBY 16

texture<float, cudaTextureType2D, cudaReadModeElementType> texRefSino;


#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__global__ void backp_kernel(
		float *image,
		int wdI,
		int nrays,
		int nangles,
		float delta,
		float dt,
		float dth)

{
  int i, j, T;
  float t, cumsum, k;
  float	x;
  float y;

  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;

  if ((i<wdI) && (j < wdI) ){

  x = (float)ini + i * delta;
  y = (float)ini + j * delta;


  cumsum = 0;

  for(k=0; k < nangles; k++)
    {

      t = x*cosf(k * dth) + y*sinf(k * dth);

      T = (float)((t + 1)/dt);

      if(T > -1 && T < nrays){
	  cumsum = cumsum + tex2D(texRefSino, k + 0.5f, T + 0.5f);
      }
    }
  image[j*wdI + i] = (cumsum*dth);
}
}

__global__ void ray_sum_kernel(float *a, float *b, size_t size, size_t size2, size_t size3)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    b[idx] = 2*a[idx];
}


extern "C" {
void Back(float *image, float *sino, int sizeImage, int nrays, int nangles){
    float* d_output;
	int size = nrays * nangles * sizeof(float);
	float dt  = 2.0/(nrays-1);
	float dth = PI/(nangles);
	float delta = (double) sqrt(2)/(sizeImage-1);


			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds = 0;

			cudaEventRecord(start);

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, nangles, nrays);

    // Copy to device memory some data located at address h_data in host memory
    cudaMemcpyToArray(cuArray, 0, 0, sino, size , cudaMemcpyHostToDevice);

    // Set texture parameters
    texRefSino.addressMode[0] = cudaAddressModeWrap;
    texRefSino.addressMode[1] = cudaAddressModeWrap;
    texRefSino.filterMode     = cudaFilterModeLinear;
  //texRefSino.normalized     = true;

    // Bind the array to the texture reference
    cudaBindTextureToArray(texRefSino, cuArray, channelDesc);

    // Allocate GPU buffers for the output image ..
    cudaMalloc(&d_output, sizeof(float) * sizeImage *sizeImage);

    //GRID N BLOCKS SIZE

    dim3 threadsPerBlock(b_TPBX,b_TPBY);
    dim3 grid((sizeImage/threadsPerBlock.x) + 1, (sizeImage/threadsPerBlock.y) + 1);


    backp_kernel<<<grid, threadsPerBlock>>>(d_output, sizeImage, nrays, nangles, delta, dt, dth);

	cudaGetLastError();

				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);

				fprintf(stderr, "%f ms\n", milliseconds);

	cudaMemcpy (image , d_output , sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost);

    cudaUnbindTexture(texRefSino);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    //cudaDeviceReset();
}}


extern "C" {
void raft_backprojection_slantstack_gpu(float *a, float *b, size_t size, size_t size2, size_t size3)
{
    float *d_a, *d_b;

    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));


    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    ray_sum_kernel <<< ceil(size / 256.0), 256 >>> (d_a, d_b, size, size2, size3);

    cudaMemcpy(b, d_b, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

}
}

