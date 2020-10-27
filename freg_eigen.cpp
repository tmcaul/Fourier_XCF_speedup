#include <iostream> 
// #include <math.h>
#include <fftw3.h>
#include <complex.h>
// #include <iomanip>
// #include <cmath>
//#include <vector>
#include <Eigen/Dense>
// #include <unsupported/Eigen/MatrixFunctions>

#include <chrono> 

using namespace std::chrono;
using namespace std;
using namespace Eigen;

#define N 40
#define red_roisize 40
#define XCF_roisize 100
#define XCF_mesh 100

// define fftshift and ifftshift 
template<class ty>
void circshift(ty *out, const ty *in, int xdim, int ydim, int xshift, int yshift)
{
  for (int i = 0; i < xdim; i++) {
    int ii = (i + xshift) % xdim;
    for (int j = 0; j < ydim; j++) {
      int jj = (j + yshift) % ydim;
      out[ii * ydim + jj] = in[i * ydim + j];
    }
  }
}
#define fftshift(out, in, x, y) circshift(out, in, x, y, (x/2), (y/2))
#define ifftshift(out, in, x, y) circshift(out, in, x, y, ((x+1)/2), ((y+1)/2))

//using Eigen::MatrixXd;

///////// Main /////////
int main()
{
    Eigen::initParallel();
    int fftw_init_threads(void);
    void fftw_plan_with_nthreads(int nthreads);

    int data_fill[red_roisize];
    int i;
    for (i=0;i<red_roisize;i++){
        data_fill[i]=i;
    }

    float roi1[N][N];
    float roi2[N][N];

    fftw_complex roi1_fft_in[N*N], roi1_fft_out[N*N], roi2_fft_in[N*N], roi2_fft_out[N*N]; /* float [2] */
    fftw_plan p, q, r;

///////// Generate Matrices & FFT them /////////
    int j;
    // generate roi 1
    for (i=0;i<N;i++)
    {
        for (j=0;j<N;j++)
        {
            roi1[i][j]=2*i*i*(i-50)+5+7*j+2;
        }
    }
    // reshape to 1D
    float *roi1_1d = (float *)roi1;

    // recast as fftw_complex
    for (i=0;i<N*N;i++)
    {
        roi1_fft_in[i][0] = roi1_1d[i];
        roi1_fft_in[i][1] = 0;
    }

    // generate roi 2
    for (i=0;i<N;i++)
    {
        for (j=0;j<N;j++)
        {
            roi2[i][j]=7*j*j*(j-5)+8*i;
        }
    }
    // reshape to 1D
    float *roi2_1d = (float *)roi2;

        // recast as fftw_complex
    for (i=0;i<N*N;i++)
    {
        roi2_fft_in[i][0] = roi2_1d[i];
        roi2_fft_in[i][1] = 0;
    }

    // FFT the ROIs
    p = fftw_plan_dft_2d(N, N, roi1_fft_in, roi1_fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    q = fftw_plan_dft_2d(N, N, roi2_fft_in, roi2_fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(q);
    fftw_destroy_plan(q);

    double cc[XCF_roisize*XCF_roisize*4];
    fftw_complex CC[XCF_roisize*XCF_roisize*4];
    r = fftw_plan_dft_c2r_2d(XCF_roisize*2, XCF_roisize*2, CC, cc, FFTW_BACKWARD);

/////////  /////////

// // TIME FROM HERE // - this is to functionalise
auto start = high_resolution_clock::now(); // WATCH FOR MEMORY LEAK

    fftw_complex f[N*N];
    for (i=0; i<N*N; i++)
    {
        f[i][0]=roi1_fft_out[i][0]*(roi2_fft_out[i][0]);
        f[i][1]=roi1_fft_out[i][1]*(-roi2_fft_out[i][1]);
    }

    // perform an fftshift on x
    float f_real[N*N];
    float f_im[N*N];
    float f_real_shifted[N*N];
    float f_im_shifted[N*N];
    for (i=0; i<N*N; i++)
        {
        f_real[i]=f[i][0];
        f_im[i]=f[i][1];
    }
    fftshift(f_real_shifted,f_real,N,N);
    fftshift(f_im_shifted,f_im,N,N);

    fftw_complex f_shifted[N*N];
    for (i=0; i<N*N; i++)
    {
        f_shifted[i][0]=f_real_shifted[i];
        f_shifted[i][1]=f_im_shifted[i];
    }

    float CC_real[XCF_roisize*2][XCF_roisize*2];
    float CC_im[XCF_roisize*2][XCF_roisize*2];

    int ind1=XCF_roisize-floor(red_roisize/2);
    int ind2=XCF_roisize+floor((red_roisize-1)/2)+1;

    // insert f into the centre of CC
    int temp=0;
    for (i=0;i<XCF_roisize*2;i++){
        for (j=0;j<XCF_roisize*2;j++){
            if (i>=ind1 && i<ind2 && j>=ind1 && j<ind2){
                CC_real[i][j]=f_shifted[temp][0];
                CC_im[i][j]=f_shifted[temp][1];
                temp+=1;
            }
            else{
                CC_real[i][j]=0;
                CC_im[i][j]=0;
            }
        }
    }

    // put CC into fftw_complex array
    // reshape to 1D
    float *CC_real_1d = (float *)CC_real;
    float *CC_im_1d = (float *)CC_im;
    for (i=0; i<XCF_roisize*XCF_roisize*4; i++)
    {
        CC[i][0]=CC_real_1d[i];
        CC[i][1]=CC_im_1d[i];
    }


    // THIS IS SLOW // - about 40-45% of runtime
    // inverse FFT CC
    fftw_execute(r);
    // normalise
    for (i=0; i<XCF_roisize*XCF_roisize*4; i++){
        cc[i]*=1./(XCF_roisize*XCF_roisize*4);
        // cc[i][1]*=1./(XCF_roisize*XCF_roisize*4);
    }
    fftw_destroy_plan(r);
    void fftw_cleanup_threads(void);

        // get maxima and locations - JUST LOOK AT REAL VALUE - replicating np.amax()
        float chi=cc[0];
        int loc=0;
        int rloc, cloc;

        for (i=0; i<XCF_roisize*XCF_roisize*4; i++){
            if (cc[i]>chi){
                chi=cc[i];
                loc=i;
            }
        }

        cloc=loc%(XCF_roisize*2);
        rloc=(loc-cloc)/(XCF_roisize*2);
        int rlocsave=rloc;
        int clocsave=cloc;

        // get shifts in the original pixel grid
        int XCF_roisize2=2*XCF_roisize;

        float row_shift;
        float col_shift;

        if(rloc>XCF_roisize){
            row_shift = rloc - XCF_roisize2;
        } else{
            row_shift = rloc;
        }

        if(cloc>XCF_roisize){
            col_shift = cloc - XCF_roisize2;
        } else{
            col_shift = cloc;
        }

        row_shift*=0.5;
        col_shift*=0.5;

        float row_shift2 = round(row_shift*XCF_mesh)/XCF_mesh;
        float col_shift2 = round(col_shift*XCF_mesh)/XCF_mesh;
        float dftshift = floor(ceil(XCF_mesh*1.5)/2);

        float roff = dftshift-row_shift2*XCF_mesh;
        float coff = dftshift-col_shift2*XCF_mesh;

        // this is an imaginary number
        std::complex<double> XCF_roisize_complex = XCF_roisize;
        std::complex<double> XCF_mesh_complex = XCF_mesh;
        std::complex<double> imag_const=-1;
        imag_const=sqrt(imag_const);
        std::complex<double> prefac=imag_const*(-2*(3.14159)/(XCF_roisize_complex*XCF_mesh_complex));

        float c1[XCF_roisize];
        for (i=0; i<XCF_roisize; i++){
            c1[i]=i;
        }
        float c_i[XCF_roisize], r_i[XCF_roisize];
        ifftshift(c_i,c1,1,XCF_roisize);
        ifftshift(r_i,c1,1,XCF_roisize);

        float dXCF_roisize=XCF_roisize;
        for (i=0;i<XCF_roisize;i++){
        c_i[i]-=floor(dXCF_roisize/2);
        r_i[i]-=floor(dXCF_roisize/2);
        }

        Eigen::MatrixXd r_i_red(1,red_roisize);
        Eigen::MatrixXd c_i_red(red_roisize,1);
        for (i=0;i<red_roisize;i++){
            r_i_red(0,i)=r_i[data_fill[i]];
            c_i_red(i,0)=c_i[data_fill[i]];
        }

        int m_length=ceil(XCF_mesh*1.5);
        Eigen::MatrixXd m1(1,m_length);
        Eigen::MatrixXd m2(m_length,1);
        for (i=0;i<m_length;i++){
            m1(0,i)=i-coff;
            m2(i,0)=i-roff;
        }

        // here to the end takes about 60% of the runtime

        Eigen::MatrixXcd kernc=prefac*c_i_red*m1;
        kernc= kernc.array().exp().matrix();
        Eigen::MatrixXcd kernr=prefac*m2*r_i_red;
        kernr=kernr.array().exp().matrix();

        Eigen::MatrixXcd kern(N,N);
        std::complex<double> comp1, comp2;
        int col;
        for (i = 0; i < N*N; i++) { 
                comp1=roi1_fft_out[i][0]+imag_const*roi1_fft_out[i][1];
                comp2=roi2_fft_out[i][0]-imag_const*roi2_fft_out[i][1];
                col=i%N;
                kern((i-col)/N,col)=comp1*comp2;
        }

        Eigen::MatrixXcd CC2=kernr*kern*kernc; //this one line takes about 40% of the runtime!

        // get maximum value and index
        int rloc_cc2, cloc_cc2;
        chi=CC2.array().abs().maxCoeff(&rloc_cc2,&cloc_cc2);

    float rloc1=rloc-dftshift-1;
    float cloc1=cloc-dftshift-1;

    float row_shift3=row_shift2+rloc1/XCF_mesh;
    float col_shift3=col_shift2+cloc1/XCF_mesh;

    // NORMALISE by getting autocorrelations of the inputs
    float bf1=0;
    float bf2=0;
    for (i=0;i<N*N;i++){
        bf1+=roi1_fft_out[i][0]*roi1_fft_out[i][0]+roi1_fft_out[i][1]*roi1_fft_out[i][1];
        bf2+=roi2_fft_out[i][0]*roi2_fft_out[i][0]+roi2_fft_out[i][1]*roi2_fft_out[i][1];
    }

    float CCmax=chi/sqrt(bf1*bf2);

        
std::cout<<CCmax<<"\n";
auto stop = high_resolution_clock::now(); 
auto duration = duration_cast<microseconds>(stop - start); 

cout << duration.count() << endl; 
}
//g++ $(pkg-config --cflags --libs fftw3) freg.cpp -o freg -Ofast -flto
//g++-9 -framework Accelerate -DEIGEN_USE_BLAS -Ofast $(pkg-config --cflags --libs fftw3 eigen3) freg_eigen.cpp -o freg_eigen -Ofast -flto -msse2 -fopenmp -DNDEBUG -Wa,-q  && ./freg_eigen