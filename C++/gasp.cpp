/* ---------------------------------------------
   Graph Attributes and Structure Pairing (GASP)
   ---------------------------------------------

Install eigen:
- Download eingen and unzip it
- sudo cp -r eigen-3.4.0/Eigen/ /usr/local/include/

Compile with:
- gcc -shared -fPIC -o C++/gasp.so C++/gasp.cpp -lstdc++

Inspiration:
- https://asiffer.github.io/posts/numpy/

------------------------------------------------------------------------- */

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

extern "C" {

  void scores(double *x,  double *y, double *as, double *at, double *bs, double *bt, int nA, int nB, int mA, int mB, double f, int nIter) {

    // --- Input mapping ---------------------------------------------------

    // --- Source-edge and target-edge matrices

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> As(as, nA, mA);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> At(at, nA, mA);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> Bs(bs, nB, mB);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> Bt(bt, nB, mB);

    // Transposition
    MatrixXd AsT = As.transpose();
    MatrixXd AtT = At.transpose();
    MatrixXd BsT = Bs.transpose();
    MatrixXd BtT = Bt.transpose();
    
    // NB: to map the transpose directly:
    // Map<Matrix<bool, Dynamic, Dynamic, RowMajor>> aS(As, nA, mA);

    // --- Score matrices

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(x, nA, nB);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> Y(y, mA, mB);

    // --- Computation -----------------------------------------------------

    for (int i=0; i<nIter; i++) {

      X = (As*Y*BsT + At*Y*BtT) / f;
      Y =  AsT*X*Bs + AtT*X*Bt;

    }
  }

  // void scores(double *x,  double *y, double *as, double *at, double *bs, double *bt, int nA, int nB, int mA, int mB, double f, int nIter) {

  //   // --- Input mapping ---------------------------------------------------

  //   // --- Source-edge and target-edge matrices

  //   Map<Matrix<double, Dynamic, Dynamic>> As(as, mA, nA);
  //   Map<Matrix<double, Dynamic, Dynamic>> At(at, mA, nA);
  //   Map<Matrix<double, Dynamic, Dynamic>> Bs(bs, mB, nB);
  //   Map<Matrix<double, Dynamic, Dynamic>> Bt(bt, mB, nB);

  //   // Transposition
  //   MatrixXd AsT = As.transpose();
  //   MatrixXd AtT = At.transpose();
  //   MatrixXd BsT = Bs.transpose();
  //   MatrixXd BtT = Bt.transpose();
    
  //   // NB: to map the transpose directly:
  //   // Map<Matrix<bool, Dynamic, Dynamic, RowMajor>> aS(As, nA, mA);

  //   // --- Score matrices

  //   Map<Matrix<double, Dynamic, Dynamic>> X(x, nB, nA);
  //   Map<Matrix<double, Dynamic, Dynamic>> Y(y, mB, mA);

  //   // --- Computation -----------------------------------------------------

  //   for (int i=0; i<nIter; i++) {

  //     // X = (As*Y*BsT + At*Y*BtT) / f;
  //     // Y =  AsT*X*Bs + AtT*X*Bt;

  //     X = (BsT*Y*As + BtT*Y*At) / f;
  //     Y =  Bs*X*AsT + Bt*X*AtT;

  //   }
  // }

}