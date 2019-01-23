#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }




__global__ void raysum(float *dev_f , float *dev_r , int wdF, int wdR, float dtheta, float dt, int nrays){

	float ini, delta, x, y, cumsum, tol, ctheta, stheta, ttheta, theta, t;
	int X, Y, i, j;

	i = blockDim.x * blockIdx.x + threadIdx.x;
	j = blockDim.y * blockIdx.y + threadIdx.y;

	if ((i<wdR) && (j < nrays) ){

	theta = i*dtheta;
	t = -1.0 + j*dt;

	tol = 1.0/sqrtf(2);
	ini = -tol;
	delta = (float) sqrtf(2)/(wdF-1);

	ctheta = cosf(theta);
	stheta = sinf(theta);
	ttheta = tanf(theta);

	if(stheta < tol){
		cumsum = 0;
	for(Y = 0; Y < wdF; Y++){
		y = ini + Y*delta;
		x = (t/ctheta - y*ttheta);
		X = (int) floorf((x - ini)/delta);
	  	if(X > -1 && X < wdF-1){
			cumsum += (dev_f[Y*wdF + (X+1)] - dev_f[Y*wdF + X])*(x - (ini + X*delta))/delta + dev_f[Y*wdF + X];

		}
	}
	dev_r[j*wdR + i] = cumsum/fabsf(ctheta);
	}
	else{
	cumsum = 0;
	for(X = 0; X < wdF; X++){
		x = ini + X*delta;
		y = (t/stheta - x/ttheta);
		Y = (int) floorf((y - ini)/delta);
		if(Y > -1 && Y < wdF-1){
			cumsum += (dev_f[(Y+1)*wdF + X] - dev_f[Y*wdF + X])*(y - (ini + Y*delta))/delta + dev_f[Y*wdF + X];

		}
	}
	dev_r[j*wdR + i] = cumsum/fabsf(stheta);
	}
	}
}



int main(int argc, char *argv[]) {
	int i, j;
	float dt, dtheta;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int sizeImage = atoi(argv[2]);
	int nrays	  = atoi(argv[3]);
	int nangles	  = atoi(argv[4]);

	int wdf = sizeImage;
	int wdr = nangles;
	FILE *fp = fopen(argv[1], "r");

	float *f;
	float *radon;
	float *dev_r = NULL;
	float *dev_f = NULL;

	unsigned int grid1, grid2;
	grid1 = (unsigned int) ceilf(((float)(nangles)/16));
	grid2 = (unsigned int) ceilf(((float)(nrays)/16));
	fprintf(stderr, "%d %d\n", grid1, grid2);

	dim3 grid(grid1, grid2, 1);
	dim3 blocks(16, 16, 1);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_f, sizeof(float)*sizeImage*sizeImage));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_r , nangles*nrays*sizeof(float) ) );

	radon = (float *)malloc(nangles*nrays*sizeof(float));
	f = (float *)malloc(sizeImage*sizeImage*sizeof(float));
	for (i = 0; i < sizeImage*sizeImage; i++)
		fscanf(fp, "%f", &f[i]);

	CUDA_CHECK_RETURN(cudaMemcpy (dev_f , f , sizeImage*sizeImage*sizeof(float) , cudaMemcpyHostToDevice));

	cudaEventRecord(start);

	dt = 2.0/(nrays-1);
	dtheta = PI/(nangles-1);
	raysum<<<grid, blocks>>>(dev_f, dev_r, wdf, wdr, dtheta, dt, nrays);



	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy (radon , dev_r , nangles*nrays*sizeof(float) , cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	fprintf(stderr, "%f ms\n", milliseconds);

	for ( i = 0; i < nrays ; i++){
		for(j=0 ; j<nangles; j++){
			fprintf(stdout, "%f ", radon[(nrays-1-i)*wdr + (nangles-1-j)]);
		}
		fprintf(stdout, "\n");
	}


	CUDA_CHECK_RETURN(cudaFree((void*) dev_r));
	CUDA_CHECK_RETURN(cudaFree((void*) dev_f));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	free(radon);
	free(f);
	fclose(fp);

	return 0;
}
