// std includes
#include <cstddef>
// local includes
#include "dev_array.h"

void restore_index_and_weights(int nlev1, int idx, float wgt, int &idx1,
                               int &idx2, float &wgt1, float &wgt2);

template <typename T>
double
vert_interp_lev_kernel(float w1, float w2, const T var1L1, const T var1L2, const T missval)
{
  if (DBL_IS_EQUAL(var1L1, missval)) w1 = 0.0f;
  if (DBL_IS_EQUAL(var1L2, missval)) w2 = 0.0f;

  // clang-format off
  if      (IS_EQUAL(w1, 0.0f) && IS_EQUAL(w2, 0.0f)) return missval;
  else if (IS_EQUAL(w1, 0.0f)) return (w2 >= 0.5f) ? var1L2 : missval;
  else if (IS_EQUAL(w2, 0.0f)) return (w1 >= 0.5f) ? var1L1 : missval;
  else                         return var1L1 * (double)w1 + var1L2 * (double)w2;
  // clang-format on
}

template <typename T>
void
vert_interp_lev(size_t gridsize, int nlev1, T missval,
                const Varray<T> &vardata1, Varray<T> &vardata2, int nlev2,
                const Varray<int> &lev_idx, const Varray<float> &lev_wgt)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx = lev_idx[ilev];
      const auto wgt = lev_wgt[ilev];
      int idx1, idx2;
      float wgt1, wgt2;
      restore_index_and_weights(nlev1, idx, wgt, idx1, idx2, wgt1, wgt2);

      // upper/lower values from input field
      auto var1L1 = &vardata1[gridsize * idx1];
      auto var1L2 = &vardata1[gridsize * idx2];

      auto var2 = &vardata2[gridsize * ilev];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
#pragma acc parallel loop
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i],
                                           missval);
        }
    }
}
