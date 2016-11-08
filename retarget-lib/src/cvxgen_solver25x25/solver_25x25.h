// Produced by CVXGEN, 2011-03-22 14:18:49 -0700.
// CVXGEN is Copyright (C) 2006-2010 Jacob Mattingley, jem@cvxgen.com.
// The code in this file is Copyright (C) 2006-2010 Jacob Mattingley.
// CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial
// applications without prior written permission from Jacob Mattingley.

// Filename: solver.h.
// Description: Header file with relevant definitions.

#pragma once
// Uncomment the next line to remove all library dependencies.
//#define ZERO_LIBRARY_MODE

#ifdef MATLAB_MEX_FILE
// Matlab functions. MATLAB_MEX_FILE will be defined by the mex compiler.
// If you are not using the mex compiler, this functionality will not intrude,
// as it will be completely disabled at compile-time.
#include "mex.h"
#else
#ifndef ZERO_LIBRARY_MODE
#include <stdio.h>
#endif
#endif

// Space must be allocated somewhere (testsolver.c, csolve.c or your own
// program) for the global variables vars, params, work and settings.
// At the bottom of this file, they are externed.

#ifndef ZERO_LIBRARY_MODE
#include <math.h>
#define pm(A, m, n) printmatrix(#A, A, m, n, 1)
#endif

struct CVXGenImageResizing25x25
{

typedef struct Params_t {
  double E[2500];
  double B[50];
  double minLengthW[1];
  double minLengthH[1];
  double imageWidth[1];
  double imageHeight[1];

} Params;

typedef struct Vars_t {
  double *st;

} Vars;

typedef struct Workspace_t {
  double h[50];
  double s_inv[50];
  double s_inv_z[50];
  double b[2];
  double q[50];
  double rhs[152];
  double x[152];
  double *s;
  double *z;
  double *y;
  double lhs_aff[152];
  double lhs_cc[152];
  double buffer[152];
  double buffer2[152];

  double KKT[1525];
  double L[1375];
  double d[152];
  double v[152];
  double d_inv[152];

  double gap;
  double optval;

  double ineq_resid_squared;
  double eq_resid_squared;

  double block_33[1];

  // Pre-op symbols.

  int converged;
} Workspace;

typedef struct Settings_t {
  double resid_tol;
  double eps;
  int max_iters;
  int refine_steps;

  int better_start;
  // Better start obviates the need for s_init and z_init.
  double s_init;
  double z_init;

  int verbose;
  // Show extra details of the iterative refinement steps.
  int verbose_refinement;
  int debug;

  // For regularization. Minimum value of abs(D_ii) in the kkt D factor.
  double kkt_reg;
} Settings;


// Function definitions in /home/jem/olsr/releases/20110218225702/lib/olsr.extra/qp_solver/solver.c:
double eval_gap(void);
void set_defaults(void);
void setup_pointers(void);
void setup_indexing(void);
void set_start(void);
void fillrhs_aff(void);
void fillrhs_cc(void);
void refine(double *target, double *var);
double calc_ineq_resid_squared(void);
double calc_eq_resid_squared(void);
void better_start(void);
void fillrhs_start(void);
long solve(void);

// Function definitions in /home/jem/olsr/releases/20110218225702/lib/olsr.extra/qp_solver/matrix_support.c:
void multbymA(double *lhs, double *rhs);
void multbymAT(double *lhs, double *rhs);
void multbymG(double *lhs, double *rhs);
void multbymGT(double *lhs, double *rhs);
void multbyP(double *lhs, double *rhs);
void fillq(void);
void fillh(void);
void fillb(void);
void pre_ops(void);

// Function definitions in /home/jem/olsr/releases/20110218225702/lib/olsr.extra/qp_solver/ldl.c:
void ldl_solve(double *target, double *var);
void ldl_factor(void);
double check_factorization(void);
void matrix_multiply(double *result, double *source);
double check_residual(double *target, double *multiplicand);
void fill_KKT(void);

// Function definitions in /home/jem/olsr/releases/20110218225702/lib/olsr.extra/qp_solver/util.c:
void tic(void);
float toc(void);
float tocq(void);
void printmatrix(char *name, double *A, int m, int n, int sparse);
double unif(double lower, double upper);
float ran1(long*idum, int reset);
float randn_internal(long *idum, int reset);
double randn(void);
void reset_rand(void);

// Function definitions in /home/jem/olsr/releases/20110218225702/lib/olsr.extra/qp_solver/testsolver.c:
int main(int argc, char **argv);
void load_default_data(void);
double eval_objv(void);


static Vars vars;
static Params params;
static Workspace work;
static Settings settings;
};