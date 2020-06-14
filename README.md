# Fourier_XCF_speedup
TPM after after Guizar-Sicairos et al - "Efficient subpixel image registration algorithms" - https://doi.org/10.1364/OL.33.000156

Small repository for fast cross correlation in C++

These functions may be useful for speeding up precise, upsampled resolution cross correlation for image shift determination.

- Uses fftw3, eigen packages (which should be installed separately).
- Compiled on OSX with openMP and BLAS. 
