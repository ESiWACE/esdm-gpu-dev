#include <math.h>
#include <algorithm>

void
dfft(double *__restrict__ real, double *__restrict__ imag, int n, int sign)
{
  // n must be a power of 2
  // sign should be 1 (FT) or -1 (reverse FT)
  int j, j1, j2;
  int bit;

  // Bit reversal part
  for (int i = j = 0; i < n; i++) // The bit pattern of i and j are reverse
    {
      if (i > j) std::swap(real[i], real[j]);
      if (i > j) std::swap(imag[i], imag[j]);

      for (bit = n >> 1; j & bit; bit >>= 1) j ^= bit;
      j |= bit;
    }

  // Danielson-Lanczos Part
  for (int step = 1; step < n; step <<= 1)
    {
      const double w_r = std::cos(M_PI / step);
      const double w_i = std::sin(M_PI / step) * sign;
      double ww_r = 1;
      double ww_i = 0;
      for (int i = 0; i < step; i++)
        {
          double temp_r, temp_i;
          for (j1 = i, j2 = i + step; j2 < n; j1 += 2 * step, j2 += 2 * step)
            {
              temp_r = ww_r * real[j2] - ww_i * imag[j2];
              temp_i = ww_r * imag[j2] + ww_i * real[j2];
              real[j2] = real[j1] - temp_r;
              imag[j2] = imag[j1] - temp_i;
              real[j1] += temp_r;
              imag[j1] += temp_i;
            }
          temp_r = ww_r;
          ww_r = ww_r * w_r - ww_i * w_i;
          ww_i = temp_r * w_i + ww_i * w_r;
        }
    }

  const double norm = 1. / std::sqrt(n);
  for (int i = 0; i < n; i++) real[i] *= norm;
  for (int i = 0; i < n; i++) imag[i] *= norm;
}

void
sfft(float *__restrict__ real, float *__restrict__ imag, int n, int sign)
{
  // n must be a power of 2
  // sign should be 1 (FT) or -1 (reverse FT)
  int j, j1, j2;
  int bit;

  // Bit reversal part
  for (int i = j = 0; i < n; i++) // The bit pattern of i and j are reverse
    {
      if (i > j) std::swap(real[i], real[j]);
      if (i > j) std::swap(imag[i], imag[j]);

      for (bit = n >> 1; j & bit; bit >>= 1) j ^= bit;
      j |= bit;
    }

  // Danielson-Lanczos Part
  for (int step = 1; step < n; step <<= 1)
    {
      const float w_r = std::cos(M_PI / step);
      const float w_i = std::sin(M_PI / step) * sign;
      float ww_r = 1;
      float ww_i = 0;
      for (int i = 0; i < step; i++)
        {
          float temp_r, temp_i;
          for (j1 = i, j2 = i + step; j2 < n; j1 += 2 * step, j2 += 2 * step)
            {
              temp_r = ww_r * real[j2] - ww_i * imag[j2];
              temp_i = ww_r * imag[j2] + ww_i * real[j2];
              real[j2] = real[j1] - temp_r;
              imag[j2] = imag[j1] - temp_i;
              real[j1] += temp_r;
              imag[j1] += temp_i;
            }
          temp_r = ww_r;
          ww_r = ww_r * w_r - ww_i * w_i;
          ww_i = temp_r * w_i + ww_i * w_r;
        }
    }

  const float norm = 1. / std::sqrt(n);
  for (int i = 0; i < n; i++) real[i] *= norm;
  for (int i = 0; i < n; i++) imag[i] *= norm;
}
