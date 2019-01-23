#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592
#define INIC -0.7071067811865475244008

#define TPBXr 16
#define TPBYr 16


texture<float, cudaTextureType2D, cudaReadModeElementType> texImage;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__global__ void Kernel(float* output, float dt, float dtheta, int sizeImage, int nrays, int nangles)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	float ini, delta, x, y, cumsum1, ctheta, stheta, ttheta, theta, t, X, Y;

    if((i < nangles) && (j < nrays))
    {
    	theta = i*dtheta;

    	ini = (float)INIC;
    	delta = (float) sqrtf(2)/(sizeImage-1);

    	ctheta =cosf(theta);
    	stheta =sinf(theta);
    	ttheta =tanf(theta);

    	if(stheta < -ini){
    		cumsum1 = 0;

    	for(Y = 0; Y < sizeImage; Y++){
				y = ini + Y*delta;
				t = -1.0 + j*dt;
				x = (t/ctheta - y*ttheta);
				X = (float)((x - ini)/delta);
				if(X >= 0 && X <= (sizeImage-1))
					cumsum1 += tex2D(texImage, X + 0.5f, Y + 0.5f);
				////////////////////////////
    		}
    		output[j*nangles + i] = delta*cumsum1/fabsf(ctheta);
    	}

    	else{
    		cumsum1 = 0;

    	for(X = 0; X < sizeImage; X++){
    			x = ini + X*delta;

    			t = -1.0 + j*dt;
    			y = (t/stheta - x/ttheta);
    			Y = (float)((y - ini)/delta);
    			if(Y >= 0 && Y <= (sizeImage-1))
    				cumsum1 += tex2D(texImage, X + 0.5f, Y + 0.5f);
    		}
    		output[j*nangles + i] = delta*cumsum1/fabsf(stheta);
    	}

    }
}



void RadonWithTexture(float* h_output, float* h_input, int sizeImage, int nrays, int nangles)
{
    float* d_output;
    float dt = 2.0/(nrays-1);
    float dtheta = PI/(nangles);
    int size = sizeImage*sizeImage*sizeof(float);

    /////// KERNEL EXECUTION TIME TEST
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds2 = 0;

			cudaEventRecord(start);
	//////////////////////////////////

    // Allocate CUDA array in device memory (phantom matrix)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, sizeImage, sizeImage);

    // Copy to device memory the phantom matrix
    cudaMemcpyToArray(cuArray, 0, 0, h_input, size , cudaMemcpyHostToDevice);

    // Set texture parameters
    texImage.addressMode[0] = cudaAddressModeBorder;
    texImage.addressMode[1] = cudaAddressModeBorder;
    texImage.filterMode     = cudaFilterModeLinear;
    /*texImage.normalized     = true;*/

    // Bind the array to the texture reference
    cudaBindTextureToArray(texImage, cuArray, channelDesc);

    // Allocate GPU buffers for the output image
    cudaMalloc(&d_output, sizeof(float) * nrays * nangles);

    // GRID and BLOCKS SIZE
    dim3 threadsPerBlock(TPBXr,TPBYr);
    dim3 grid((nangles/threadsPerBlock.x) + 1, (nrays/threadsPerBlock.y) + 1);

    //KERNEL EXECUTION
    Kernel<<<grid, threadsPerBlock>>>(d_output, dt, dtheta, sizeImage, nrays, nangles);
    cudaDeviceSynchronize();

    /////// PRINT KERNEL EXECUTION TIME
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds2, start, stop);
			fprintf(stderr, "%f ms\n", milliseconds2);
    ///////////////////////////////////

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_output, d_output, sizeof(float) * nrays * nangles, cudaMemcpyDeviceToHost);

    // unbind texture from buffer
    cudaUnbindTexture(texImage);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    cudaDeviceReset();
}

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
int main(int argc, char *argv[]) {
	int i, j;

	FILE *fp = fopen(argv[1], "r");
	int sizeImage = atoi(argv[2]);
	int nrays	  = atoi(argv[3]);
	int nangles	  = atoi(argv[4]);

	float *f;
	float *radon;

	radon = (float *)malloc(nangles*nrays*sizeof(float));
	f = (float *)malloc(sizeImage*sizeImage*sizeof(float));
	for (i = 0; i < sizeImage*sizeImage; i++)
		(void)fscanf(fp, "%f", &f[i]);


	RadonWithTexture(radon, f, sizeImage, nrays, nangles);


	for ( i = 0; i < nrays ; i++){
		for(j=0 ; j<nangles; j++){
			fprintf(stdout, "%f ", radon[(nrays-1-i)*nangles + (nangles-1-j)]);
		}
		fprintf(stdout, "\n");
	}

		free(radon);
		free(f);
		fclose(fp);

		return 0;
}
