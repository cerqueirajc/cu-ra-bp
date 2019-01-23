#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define MAX_NTHREADS 20000

#define MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define MAX( x, y ) ( ( ( x ) > ( y ) ) ? ( x ) : ( y ) )
#define SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )
#define PI 3.1415926535897932384626433832795

typedef struct{

  double *phantom;
  double *radon;
  int size, nviews, nrays;
  int nthread;
  int colIndex[2];  
  
}param_t;

void *slantstack_loop(void *t);

double eval_rayintegral(double *phantom,
			int size,
			int nviews,
			int nrays,
			int j,
			int k);

/*========================================================*/

/*!
 * \brief Radon transform: slant stack
 * \param phantom image
 * \param nthreads number of threads
 */

void funcradon(double *phantom,
	   double *radon,
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
  
  e = (int) floor((nviews*nrays)/nthreads);
      
  for(n = 0; n < nthreads+1; n++) 
    {
      param[n].phantom = phantom;
      param[n].radon   = radon;
      param[n].size    = size; 
      param[n].nviews  = nviews;
      param[n].nrays   = nrays;
      param[n].nthread = n;      
      
      param[n].colIndex[0] = e * n;
      param[n].colIndex[1] = (n+1) * e;
      
      rc = pthread_create(&thread[n], &attr, slantstack_loop, (void *)&param[n]);
    }
  
  // Free attribute and wait for the other threads
  pthread_attr_destroy(&attr);
  for(n = 0; n < nthreads+1; n++) 
    {
      rc = pthread_join(thread[n], &status);
    }  

  rc++;

}

/*=========== PRIVATE FUNCTIONS ========================*/

void *slantstack_loop(void *t)
{
  int w, j, k, ndata;
  param_t *param;
  double sum;
  
  param = (param_t *)t;
  
  ndata = param->nviews * param->nrays;

  for(w = param->colIndex[0]; w < MIN(param->colIndex[1],ndata); w++)
    {
      j = w/param->nrays;
      k = w%param->nrays;

      sum = eval_rayintegral(param->phantom, 
			     param->size,
			     param->nviews, 
			     param->nrays, j, k);
      
      param->radon[param->nrays-1-k + j * param->nrays] = sum;
    }  
  
  pthread_exit(NULL);
}


/*=====================================================*/

double eval_rayintegral(double *phantom,
			int size,
			int nviews,
			int nrays,
			int j,
			int k)
{
  int i, m;
  double sum, cost, sint, tant;
  double x, xmin, xmax, dx;
  double y, ymin, ymax, dy;
  double theta, th0, thf, dth;
  double t, tmin, tmax, dt; 
  double fsint, fcost, tol, value, Delta;
  
  tol  = 1.0/sqrt(2);
  
  // image corners
  xmin = -tol;
  xmax = tol;

  ymin = -tol;
  ymax = tol;

  //sampling distance for phantom
  dx = (xmax - xmin)/(size - 1);
  dy = (ymax - ymin)/(size - 1);
  
  //angle 
  th0 = 0;
  thf = PI;
  dth = (thf-th0)/(nviews);

  theta = th0 + j * dth;

  cost  = cos(theta);
  sint  = sin(theta);
  tant  = tan(theta);  
  fsint = fabs(sint);
  fcost = fabs(cost);
  
  //ray @ angle
  
  tmin = -1.0;
  tmax = 1.0;
  dt = (tmax - tmin)/(nrays - 1);

  t = tmin + k * dt;
   
  if(fcost < tol) 
    {
      sum   = 0;
      Delta = (dy/fsint);

      for(i=0; i < size; i++)
	{
	  x = (double) (xmin + i * dx);
	  y = (double) (t/sint - x/tant);

	  m = (int) floor ((y-ymin)/dy);

	  if(m > -1 && m < size-1)	  
	    {
	      sum += (phantom[i+(m+1)*size]-phantom[i + m*size])*(y-(ymin+m*dy))/dy + phantom[i + m*size];
	    }
	}
      value = sum*Delta;
    } 
  else
    {
      sum   = 0;
      Delta = (dx/fcost);

      for(i=0; i < size; i++)
	{
	  y = (double) (xmin + i * dy); 
	  x = (double) (t/cost - y*tant);
	  
	  m = (int) floor ( (x-xmin)/dx);

	  if(m > -1 && m < size-1)	  
	    {
	      sum += (phantom[m+1+i*size]-phantom[m+i*size])*(x-(xmin+m*dx))/dx + phantom[m + i*size];
	    }
	  
	}
      value = sum*Delta;
    }
  
  return value;
}
/////////////////////////////////


int main(int argc, char *argv[]) {
	int i, j;

	int sizeImage = atoi(argv[2]);
	int nrays	  = atoi(argv[3]);
	int nangles	  = atoi(argv[4]);

	int nthreads = 16;

	double *f;
	double *radon;
	FILE *fp = fopen(argv[1], "r");

		struct timeval  tv1, tv2;

	radon = (double *)malloc(nangles*nrays*sizeof(double));
	f = (double *)malloc(sizeImage*sizeImage*sizeof(double));
	for (i = 0; i < sizeImage*sizeImage; i++)
		fscanf(fp, "%lf", &f[i]);

gettimeofday(&tv1, NULL);

	funcradon(f, radon, sizeImage, nrays, nangles, nthreads);

gettimeofday(&tv2, NULL);

		fprintf(stderr, "%f \n",  (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
        						 (double) (tv2.tv_sec - tv1.tv_sec));


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
