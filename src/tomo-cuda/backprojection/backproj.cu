#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592


#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



__global__ void anglesum(float *sino,
		float *image,
		int nangles,
		int nrays,
		int wdI,
		float delta,
		float ini)

{
  int i, j, T, k;
  float dt, th, dth, t, cost, sint, cumsum;
  float	x;
  float y;

  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;

  if ((i<wdI) && (j < wdI) ){

  x = ini + i * delta;
  y = ini + j * delta;



  cumsum = 0;

  dt  = 2.0/(nrays-1);
  dth = PI/(nangles);

  for(k=0; k < nangles; k++)
    {
      th = k*dth;

      cost  = cos(th);
      sint  = sin(th);

      t = x*cost + y*sint;

      T = (int) floor((t + 1)/dt);

      if(T > -1 && T < nrays-1)
	{
	  cumsum = cumsum + (sino[nangles*(T+1) + k]-sino[nangles*T + k])*(t-(-1.0 + T*dt))/dt + sino[nangles*T + k];
	}
    }
  image[j*wdI + i] = (cumsum*dth);
}
}


int main(int argc, char *argv[]) {
	int i, j;
	float delta, ini;

	int sizeImage = atoi(argv[2]);
	int nrays     = atoi(argv[3]);
	int nangles   = atoi(argv[4]);

	FILE *fp=fopen(argv[1], "r");


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	float *dev_i = NULL;
	float *dev_s = NULL;
	float *image;
	float *sino;

	// GRID SIZE
	unsigned int gridp;
	gridp = (unsigned int) ceilf(((float)(sizeImage)/16));
	dim3 grid(gridp, gridp, 1);
	dim3 blocks(16, 16, 1);

	// MALLOC AND CUDA COPY
	image = (float *)malloc(sizeImage*sizeImage*sizeof(float));
	sino = (float *)malloc(nangles*nrays*sizeof(float));
	for (i = 0; i < nangles*nrays; i++)
		fscanf(fp, "%f", &sino[i]);


	cudaEventRecord(start);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_i, sizeof(float)*sizeImage*sizeImage));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_s , nangles*nrays*sizeof(float) ) );
	CUDA_CHECK_RETURN(cudaMemcpy (dev_s , sino , nrays*nangles*sizeof(float) , cudaMemcpyHostToDevice));
	//////////////////////

	// BACK PROJECTION
	ini = -sqrt(2)/2;

	delta = (double) sqrt(2)/(sizeImage-1);

	anglesum<<<grid, blocks>>>(dev_s, dev_i, nangles, nrays, sizeImage, delta, ini);
	// BACK PROJECTION FINISHED

	// COPY RESULTS
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaMemcpy (image , dev_i , sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost) );

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	fprintf(stderr, "%f ms\n", milliseconds);

	  for(i=0; i< sizeImage; i++)
	    {
	      for(j=0; j< sizeImage; j++)
		{
		  fprintf(stdout, "%f ", image[sizeImage*(sizeImage-1-i) + j]);
		}
	      fprintf(stdout, "\n");
	    }
	  //////////////

	  // FREE MEMORY
	free(sino);
	free(image);
	CUDA_CHECK_RETURN(cudaFree((void*) dev_s));
	CUDA_CHECK_RETURN(cudaFree((void*) dev_i));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
