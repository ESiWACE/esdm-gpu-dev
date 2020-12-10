// std includes
#include <iostream>

// local includes
#include "dev_interpolation.h"
int main(int argc, char ** argv){
    size_t gridsize = 8000000;
    size_t nLev1 = 8;
    size_t nLev2 = nLev1;
    auto vardata1 = Varray<float>(gridsize*nLev1,1);
    auto vardata2 = Varray<float>(gridsize*nLev2,1);
    auto lev_idx = Varray<int>(nLev1,1);
    auto lev_wgt = Varray<float>(nLev1,1);
    float misval = 0.0f;
    vert_interp_lev(gridsize, nLev1, misval,vardata1,vardata2, nLev2, lev_idx,lev_wgt);
}
