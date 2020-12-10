#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <utility>         // for std::pair

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

// functions for vert_interp_lev

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

#endif // FUNCTIONS_H
