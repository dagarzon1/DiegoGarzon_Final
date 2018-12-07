#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
double gsf(double x, double mu, double s);

int main(int argc,char ** argv)
{
    #pragma omp parallel
    {
	srand(time(NULL));
	int N = 1000;
	double mu = 0.0;
	double s = 1.0;
	double * gs = new double[N];
	int mn = mu - 4*s;
	int mx = mu + 4*s;
	gs[0]=0.1;

	for(int i=1;i<N;i++)
	{
		
		double t = double(rand()) / double(RAND_MAX) ;
		t = ( mx-mn ) * t + mn;
		double alfa = double( rand() ) / double(RAND_MAX);
		double gs_i = gs[i-1] +  t;
		double r = gsf(gs_i,mu,s) / gsf(gs[i-1],mu,s);
		if (alfa < r)
		{
			gs[i]=gs_i;
		}
		else
		{
			gs[i]=gs[i-1];
		}
	}
    int id = omp_get_thread_num();
    char buffer [50];
    ofstream f_w;
    stringstream ss;
    sprintf(buffer,"%d",id);
    ss << buffer;
    string name;
    ss >> name;
    string fil = "sample_" + name + ".txt";
    f_w.open(fil.c_str());
    for(int i=0;i<N;i++)
    {
        f_w << gs[i] << endl ;
    }
        f_w.close();
    }
    
return 0;
}
double gsf(double x, double mu, double s)
{
	double pi = acos(-1);
	return 1.0/( s * sqrt(2 * pi) ) * exp( - ( x - mu ) * ( x - mu) / ( 2.0 * s * s ) );
	
}