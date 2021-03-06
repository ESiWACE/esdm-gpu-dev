#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <utility>         // for std::pair

#define FFT_INTERNAL 0
#define FFT_FFTW 1

// functions for fft
//
void sfft(float *__restrict__ real, float *__restrict__ imag, int n, int sign);
void dfft(double *__restrict__ real, double *__restrict__ imag, int n, int sign);
//void CPUs_fft_fftw(const size_t fftsize, float *array);
//void CPUd_fft_fftw(const size_t fftsize, double *array);
//void GPUs_fft_fftw(const size_t fftsize, float *array);
//void GPUd_fft_fftw(const size_t fftsize, double *array);

// functions for varrayMinMaxMV

std::pair<float, float>
CPUs_varrayMinMaxMV(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
CPUd_varrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals);
std::pair<float, float>
GPUs_varrayMinMaxMV(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
GPUd_varrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals);

// noisnan versions
//
std::pair<float, float>
CPUs_varrayMinMaxMV_noisnan(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
CPUd_varrayMinMaxMV_noisnan(const size_t len, const double *array, const double missval, size_t &nvals);
std::pair<float, float>
GPUs_varrayMinMaxMV_noisnan(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
GPUd_varrayMinMaxMV_noisnan(const size_t len, const double *array, const double missval, size_t &nvals);

// nomissval versions
//
std::pair<float, float>
CPUs_varrayMinMaxMV_nomissval(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
CPUd_varrayMinMaxMV_nomissval(const size_t len, const double *array, const double missval, size_t &nvals);
std::pair<float, float>
GPUs_varrayMinMaxMV_nomissval(const size_t len, const float *array, const float missval, size_t &nvals);
std::pair<double, double>
GPUd_varrayMinMaxMV_nomissval(const size_t len, const double *array, const double missval, size_t &nvals);

// functions for vert_interp_lev
//
void
CPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
CPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

void
GPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

// noisnan versions
//
void
CPUs_vert_interp_lev_noisnan(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
CPUd_vert_interp_lev_noisnan(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

void
GPUs_vert_interp_lev_noisnan(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
GPUd_vert_interp_lev_noisnan(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

// nomissval versions
//
void
CPUs_vert_interp_lev_nomissval(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
CPUd_vert_interp_lev_nomissval(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

void
GPUs_vert_interp_lev_nomissval(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2);

void
GPUd_vert_interp_lev_nomissval(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2);

#endif // FUNCTIONS_H
