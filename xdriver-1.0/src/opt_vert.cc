#include <cstddef>

#define MAXGRID 8192
#define MAXLEV 8

void
GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
#pragma acc data copyin(vardata1[:MAXGRID*MAXLEV],vardata2[:MAXGRID*MAXLEV])
#pragma acc data copyin(lev_idx1[:MAXGRID],lev_idx2[:MAXGRID],lev_wgt1[:MAXGRID],lev_wgt2[:MAXGRID])
  
#pragma acc parallel loop
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      double wgt1 = lev_wgt1[ilev];
      double wgt2 = lev_wgt2[ilev];
      const int lev1 = lev_idx1[ilev];
      const int lev2 = lev_idx2[ilev];
#pragma acc data copyin(vardata1[MAXGRID*lev1:MAXGRID*lev1+MAXGRID]) 
#pragma acc data copyin(vardata2[MAXGRID*lev2:MAXGRID*lev2+MAXGRID]) 
#pragma acc data copyout(vardata2[MAXGRID*ilev:MAXGRID*ilev+MAXGRID]) 

#pragma acc loop independent
      for (size_t i = 0; i < gridsize; ++i)
        {
          if (vardata1[i+gridsize*lev1]!=missval) wgt1 = 0;
          if (vardata2[i+gridsize*lev2]!=missval) wgt2 = 0;

          if ((wgt1!=0) && (wgt2!=0))
            {
              vardata2[i+gridsize*ilev] = missval;
            }
          else if (wgt1!=0)
            {
              vardata2[i+gridsize*ilev] = (wgt2 >= 0.5) ? vardata2[i+gridsize*lev2] : missval;
            }
          else if (wgt2!=0)
            {
              vardata2[i+gridsize*ilev] = (wgt1 >= 0.5) ? vardata1[i+gridsize*lev1] : missval;
            }
          else
            {
              vardata2[i+gridsize*ilev] = vardata1[i+gridsize*lev1] * wgt1 
                                        + vardata2[i+gridsize*lev2] * wgt2;
            }
        }
    }
}
