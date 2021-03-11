#if defined(_OPENMP)
#include <omp.h>
#define TIMER_FUNCTION omp_get_wtime()
#else
#define TIMER_FUNCTION (double)clock()/CLOCKS_PER_SEC
#endif
