#include "dev_interpolation.h"

constexpr int BottomLevel = 32000;
constexpr int TopLevel    = 32001;

void
restore_index_and_weights(int nlev1, int idx, float wgt, int &idx1, int &idx2, float &wgt1, float &wgt2)
{
  if (idx == BottomLevel)
    {
      idx1 = 0;
      idx2 = 0;
      wgt1 = 0.0f;
      wgt2 = wgt;
    }
  else if (idx == TopLevel)
    {
      idx1 = nlev1 - 1;
      idx2 = nlev1 - 1;
      wgt1 = wgt;
      wgt2 = 0.0f;
    }
  else
    {
      idx1 = (idx < 0) ? -idx : idx;
      idx2 = (idx < 0) ? idx1 - 1 : idx1 + 1;
      wgt1 = wgt;
      wgt2 = 1.0f - wgt;
    }
  // printf("%d %d %g %g\n", idx1, idx2, wgt1, wgt2);
}

//  1D vertical interpolation

