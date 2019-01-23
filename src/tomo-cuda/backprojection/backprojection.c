#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define MAX_NTHREADS 20000
#define MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define MAX( x, y ) ( ( ( x ) > ( y ) ) ? ( x ) : ( y ) )
#define SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )
#define PI 3.1415926535897932384626433832795

typedef struct{

  double *sinogram;
  double *backprojection;
  int size;
  int nviews;
  int nrays;    
  int nthread;
  int colIndex[2];  
  
}param_t;

void *back_slantstack_loop(void *t);

double eval_anglesum_slantstack(double *sinogram,
				int size,
				int nrays,
				int nviews,				
				int j,
				int k);

/*==============================================================*/

/*!
 * \brief Backprojection transform: slant stack
 * \param sinogram image
 * \param nthreads number of threads 
 */

void backprojection(double *sinogram,
		    double *backprojection,
		    int size,
		    int nrays,
		    int nviews,
		    int nthreads)
{
  pthread_t thread[MAX_NTHREADS];
  pthread_attr_t attr;
  int e, n, rc;    
  param_t param[MAX_NTHREADS];  
  void *status;

  // Initialize and set thread detached attribute
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  e = (int) floor((size*size)/nthreads);;
      
  for(n = 0; n < nthreads+1; n++) 
    {
      param[n].size           = size;
      param[n].nviews         = nviews;
      param[n].nrays          = nrays;
      param[n].sinogram       = sinogram;
      param[n].backprojection = backprojection; 
      param[n].nthread        = n;      
      
      param[n].colIndex[0]   = e * n;
      param[n].colIndex[1]   = (n+1) * e;
      
      rc = pthread_create(&thread[n], 
			  &attr, 
			  back_slantstack_loop, 
			  (void *)&param[n]);
    }

  // Free attribute and wait for the other threads
  pthread_attr_destroy(&attr);
  for(n = 0; n < nthreads+1; n++) 
    {
      rc = pthread_join(thread[n], &status);
    }  

  rc = rc + 1; 	
}

/*==============================================================*/

void *back_slantstack_loop(void *t)
{
  int w, j, k, size, ndata;
  param_t *param;
  double sum;
  
  param = (param_t *)t;
  
  size = param->size;

  ndata = size*size;
  
  for(w = param->colIndex[0]; w < MIN(param->colIndex[1],ndata); w++)
    {
      j = w/size;
      k = w%size;
      
      sum = eval_anglesum_slantstack(param->sinogram, 
				     param->size,
				     param->nrays,
				     param->nviews, 
				     j, k);	  
      
      param->backprojection[k + j*size] = sum;
    }  
  
  pthread_exit(NULL);
}

/*=======================================================*/

double eval_anglesum_slantstack(double *sinogram,
				int size,
				int nrays,
				int nviews,
				int j,
				int k)
{
  int i, m, n;
  double cost, sint, sum, s, tol;
  double th0, thf, th, dth;
  double t, tmin, tmax, dt;
  double x, xmin, xmax, dx;
  double y, ymin, ymax, dy;
  int BPindex[4]; 
  
  tmin = -1.0;
  tmax = 1.0;

  dt = (tmax-tmin)/(nrays-1);

  th0 = 0;
  thf = PI;
  
  dth = (thf-th0)/(nviews);

  tol = 1.0/sqrt(2);

  xmin = -tol;
  xmax = tol;
  dx   = (xmax - xmin)/(size - 1);

  ymin = -tol;
  ymax = tol;
  dy   = (ymax - ymin)/(size - 1);
  
  x = xmin + k * dx;

  y = ymin + (size-1-j) * dy;

  // stacking 
  
  BPindex[0] = 0;
  BPindex[1] = nviews/4;
  BPindex[2] = nviews/2;
  BPindex[3] = 3*nviews/4;
  
  sum = 0;
  
  for(i=0; i < nviews/4; i++)
    {
      s = 0;

      for (m = 0; m < 4; m++)
	{
	  j = (i + BPindex[m]) % nviews; 

	  th = th0 + j*dth;
	  
	  cost  = cos(th);
	  sint  = sin(th);
	  
	  t =  x*cost + y*sint;
	  
	  n = (int) floor ( (t-tmin)/dt );  
	  
	  if(n > -1 && n < nrays-1)
	    {
	      s += (sinogram[n+1+j*nrays]-sinogram[n+j*nrays])*(t-(tmin+(n)*dt))/dt + sinogram[n+j*nrays];
	    }
	}
      
      sum += s;
    }
 
  return sum*dth;
}


////////////////
int main(int argc, char *argv[]) {
	int i, j;

	int sizeImage = atoi(argv[2]);
	int nrays     = atoi(argv[3]);
	int nangles   = atoi(argv[4]);

	int nthreads = 8;

	FILE *fp=fopen(argv[1], "r");

		struct timeval  tv1, tv2;

	double *image;
	double *sino;

	image = (double *)malloc(sizeImage*sizeImage*sizeof(double));
	sino = (double *)malloc(nangles*nrays*sizeof(double));
	for (i = 0; i < nangles*nrays; i++)
		fscanf(fp, "%lf", &sino[i]);


//////
gettimeofday(&tv1, NULL);

	backprojection(sino,
		    image,
		    sizeImage,
		    nrays,
		    nangles,
		    nthreads);

gettimeofday(&tv2, NULL);

		fprintf(stderr, "%f \n",  (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
        						 (double) (tv2.tv_sec - tv1.tv_sec));

/////

	  for(i=0; i< sizeImage; i++)
	    {
	      for(j=0; j< sizeImage; j++)
		{
		  fprintf(stdout, "%f ", image[sizeImage*(sizeImage-1-i) + j]);
		}
	      fprintf(stdout, "\n");
	    }




	free(sino);
	free(image);

	return 0;
}

