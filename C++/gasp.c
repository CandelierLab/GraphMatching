/* ---------------------------------------------
   Graph Attributes and Structure Pairing (GASP)
   ---------------------------------------------

Compile with:
- gcc -shared -fPIC -o C++/gasp.so C++/gasp.c

Inspiration:
- https://asiffer.github.io/posts/numpy/

------------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>

void scores(double *X,  double *Y, int *eA, int *eB, int nA, int nB, int mA, int mB, double f, int nIter) {

  // --- Index lists -----------------------------------------------------

  int N = 2*mA*mB;
  int I[N];
  int J[N];

  int h = 0;
  int k = 0;

  for (int i=0; i<mA; i++) {

    for (int j=0; j<mB; j++) {

      // --- Sources

      I[h] = k;
      J[h] = eA[2*i]*nB + eB[2*j];

      // Update
      h++;

      // --- Targets

      I[h] = k;
      J[h] = eA[2*i+1]*nB + eB[2*j+1];

      // Update
      h++;
      k++;

    }
  }

  // --- Computation -----------------------------------------------------

  for (int iter=0; iter<nIter; iter++) {

    // Update X
    memset(X, 0, nA*nB*sizeof(double));
    for (int i=0; i<N; i++) { X[J[i]] += Y[I[i]]; }

    // Normalize X
    for (int i=0; i<nA*nB; i++) { X[i]/= f; }

    // Update Y
    memset(Y, 0, mA*mB*sizeof(double));
    for (int i=0; i<N; i++) { Y[I[i]] += X[J[i]]; }

  }

}
