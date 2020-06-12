#include <iostream> 
#include <math.h>
#include <fftw3.h>
#include <complex.h>
#include <iomanip>
#include <cmath>
#include <vector>

#include <chrono> 
using namespace std::chrono;

using namespace std;

#define N 46

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

///////// Main /////////
int main()
{

    /// params to edit
    int red_roisize = 46;
    int XCF_roisize = 150;

    int data_fill[red_roisize];
    int i;
    for (i=0;i<red_roisize;i++){
        data_fill[i]=i;
    }

    int XCF_mesh = 150;

    double roi1[N][N];
    double roi2[N][N];

    fftw_complex roi1_fft_in[N*N], roi1_fft_out[N*N], roi2_fft_in[N*N], roi2_fft_out[N*N]; /* double [2] */
    fftw_plan p, q, r;

///////// Generate Matrices /////////
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
    double *roi1_1d = (double *)roi1;

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
    double *roi2_1d = (double *)roi2;

        // recast as fftw_complex
    for (i=0;i<N*N;i++)
    {
        roi2_fft_in[i][0] = roi2_1d[i];
        roi2_fft_in[i][1] = 0;
    }

////////////////////////


    // FFT the ROIs
    p = fftw_plan_dft_2d(N, N, roi1_fft_in, roi1_fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    q = fftw_plan_dft_2d(N, N, roi2_fft_in, roi2_fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(q);
    fftw_destroy_plan(q);


// // TIME FROM HERE //
// auto start = high_resolution_clock::now(); // WATCH FOR MEMORY LEAK

// int iterations;
// for(iterations=0;iterations<1;iterations++){
    // pointwise multiply with conjugate
    fftw_complex f[N*N];
    for (i=0; i<N*N; i++)
    {
        f[i][0]=roi1_fft_out[i][0]*(roi2_fft_out[i][0]);
        f[i][1]=roi1_fft_out[i][1]*(-roi2_fft_out[i][1]);
    }

    // perform an fftshift on x
    double f_real[N*N];
    double f_im[N*N];
    double f_real_shifted[N*N];
    double f_im_shifted[N*N];
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

    double CC_real[XCF_roisize*2][XCF_roisize*2];
    double CC_im[XCF_roisize*2][XCF_roisize*2];

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
    double *CC_real_1d = (double *)CC_real;
    double *CC_im_1d = (double *)CC_im;
    fftw_complex CC[XCF_roisize*XCF_roisize*4];
    for (i=0; i<XCF_roisize*XCF_roisize*4; i++)
    {
        CC[i][0]=CC_real_1d[i];
        CC[i][1]=CC_im_1d[i];
    }

    // inverse FFT CC
    fftw_complex cc[XCF_roisize*XCF_roisize*4];
    r = fftw_plan_dft_2d(XCF_roisize*2, XCF_roisize*2, CC, cc, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(r);
    // normalise
    for (i=0; i<XCF_roisize*XCF_roisize*4; i++){
        cc[i][0]*=1./(XCF_roisize*XCF_roisize*4);
        cc[i][1]*=1./(XCF_roisize*XCF_roisize*4);
    }
    fftw_destroy_plan(r);

    ////////////////////////////// testing //////////////////////////////
    //cc[34][0]=1234; //< this will need adjusting dep on parameters - check against jupyter notebook
        ////////////////////////////// //////////////////////////////


    // get maxima and locations - JUST LOOK AT REAL VALUE - replicating np.amax()
    double chi=cc[0][0];
    int loc=0;
    int rloc, cloc;

    for (i=0; i<XCF_roisize*XCF_roisize*4; i++){
        if (cc[i][0]>chi){
            chi=cc[i][0];
            loc=i;
        }
    }

    cloc=loc%(XCF_roisize*2);
    rloc=(loc-cloc)/(XCF_roisize*2);
    int rlocsave=rloc;
    int clocsave=cloc;

    // get shifts in the original pixel grid
    int XCF_roisize2=2*XCF_roisize;

    double row_shift;
    double col_shift;

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

    double row_shift2 = round(row_shift*XCF_mesh)/XCF_mesh;
    double col_shift2 = round(col_shift*XCF_mesh)/XCF_mesh;
    double dftshift = floor(ceil(XCF_mesh*1.5)/2);

    double roff = dftshift-row_shift2*XCF_mesh;
    double coff = dftshift-col_shift2*XCF_mesh;

    // this is an imaginary number
    std::complex<double> XCF_roisize_complex = XCF_roisize;
    std::complex<double> XCF_mesh_complex = XCF_mesh;
    std::complex<double> imag_const=-1;
    imag_const=sqrt(imag_const);
    std::complex<double> prefac=imag_const*(-2*(3.14159)/(XCF_roisize_complex*XCF_mesh_complex));

    double c1[XCF_roisize];
    for (i=0; i<XCF_roisize; i++){
        c1[i]=i;
    }
    double c_i[XCF_roisize], r_i[XCF_roisize];
    ifftshift(c_i,c1,1,XCF_roisize);
    ifftshift(r_i,c1,1,XCF_roisize);

    double dXCF_roisize=XCF_roisize;
    for (i=0;i<XCF_roisize;i++){
    c_i[i]-=floor(dXCF_roisize/2);
    r_i[i]-=floor(dXCF_roisize/2);
    }

    double r_i_red[red_roisize], c_i_red[red_roisize];
    for (i=0;i<red_roisize;i++){
        r_i_red[i]=r_i[data_fill[i]];
        c_i_red[i]=c_i[data_fill[i]];
    }

    int m_length=ceil(XCF_mesh*1.5);
    int m1[m_length], m2[m_length];
    for (i=0;i<m_length;i++){
        m1[i]=i-coff;
        m2[i]=i-roff;
    }

    // arg1 = c_i @ m1
    // kernc = exp(prefac*arg1)
    std::complex<double> arg;
    std::complex<double> kernc[XCF_roisize][m_length];
    for (i = 0; i < XCF_roisize; i++) { 
        for (j = 0; j < m_length; j++) { 
            arg = c_i[i] *  m1[j];
            kernc[i][j]=exp(prefac*arg);
        } 
    } 

    // arg2 = m2 @ r_i
    // kernr = exp(prefac*arg2)
    std::complex<double> kernr[m_length][XCF_roisize];
    for (i = 0; i < m_length; i++) { 
        for (j = 0; j < XCF_roisize; j++) { 
            arg = m2[i] *  c_i[j];
            kernr[i][j]=exp(prefac*arg);
        } 
    } 

    // kern = ROI_ref * conj(ROI_test);
    std::complex<double> kern[N][N];
    std::complex<double> comp1;
    std::complex<double> comp2;
    int col;
    for (i = 0; i < N*N; i++) { 
            comp1=roi1_fft_out[i][0]+imag_const*roi1_fft_out[i][1];
            comp2=roi2_fft_out[i][0]-imag_const*roi2_fft_out[i][1];
            col=i%N;
            kern[(i-col)/N][col]=comp1*comp2;
    } 

    // need to do CC2=kernr@kern@kernc
    // kern@kernc
    int k;
    std::complex<double> k2[N][m_length];
    for (i = 0; i < N; i++){ 
        for (j = 0; j < m_length; j++){ 
            k2[i][j] = 0; 
            for (k = 0; k < XCF_roisize; k++){
                k2[i][j] += kern[i][k] * kernc[k][j];
            }
        } 
    }
    //kernr@k2
    // also grab locations and maxima
    chi=0;
    loc=0;
    std::complex<double> CC2[m_length][m_length];
    for (i = 0; i < m_length; i++){ 
        for (j = 0; j < m_length; j++){ 
            CC2[i][j] = 0; 
            for (k = 0; k < N; k++){
                CC2[i][j] += kernr[i][k] * k2[k][j];
            }
            CC2[i][j]=conj(CC2[i][j]);

            // keep a running maximum
            if(abs(CC2[i][j])>chi){
                chi=abs(CC2[i][j]);
                rloc=i;
                cloc=j;
            }
        } 
    } 

    double rloc1=rloc-dftshift-1;
    double cloc1=cloc-dftshift-1;

    double row_shift3=row_shift2+rloc1/XCF_mesh;
    double col_shift3=col_shift2+cloc1/XCF_mesh;

    // NORMALISE by getting autocorrelations of the inputs
    double bf1=0;
    double bf2=0;
    for (i=0;i<N*N;i++){
        bf1+=roi1_fft_out[i][0]*roi1_fft_out[i][0]+roi1_fft_out[i][1]*roi1_fft_out[i][1];
        bf2+=roi2_fft_out[i][0]*roi2_fft_out[i][0]+roi2_fft_out[i][1]*roi2_fft_out[i][1];
    }

    double CCmax=chi/sqrt(bf1*bf2);

    // get maxima and locations - JUST LOOK AT REAL VALUE - replicating np.amax()
    // double chi=cc[0][0];
    // int loc=0;
    // int rloc, cloc;

    // for (i = 0; i < XCF_roisize*XCF_roisize*4; i++) 
    // { 
    //     cout << CC[i][0] << " "; 
    //     cout << "\n"; 
    // } 

    // print matrix to the console
    // for (i = 0; i < m_length; i++) 
    // { 
    //     for (j = 0; j < m_length; j++) 
    //     cout << CC2[i][j] << " "; 
    //     cout << "\n"; 
    // }
// }
// auto stop = high_resolution_clock::now(); 
// auto duration = duration_cast<milliseconds>(stop - start); 
// cout << duration.count()/(iterations+1) << endl; 
    // cout<<CCmax<<"\n";
    // cout<<row_shift3<<"\n";
    // cout<<col_shift3<<"\n";

    // return 0;
}
//g++ $(pkg-config --cflags --libs fftw3) freg.cpp -o freg && ./freg