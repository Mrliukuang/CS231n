#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    // input variable: cols, out_size, filter_size, stride
    // output variable: matrix of out_size=[H, W, C, N]
    // e.g. im = col_2_im_c(cols, [4,4,2,2], [2,2], 2);
    #define M_IN prhs[0]
    #define M_OUT plhs[0]
    #define A(i, j) A[i + j*cols_H]  // starts from 0
    #define B(h, w, c, n) B[h + w*H + c*H*W + n*C*H*W]
    
    double *A = mxGetPr(M_IN);  // A point to cols of size [C*filter_h*filter_w, N*HH*WW]
    double *B;                  // M_OUT data ptr
    
    int h, w, c, n;   // variables for loops
    int hh, ww;
    int i, j;
    
    double *out_sz = mxGetPr(prhs[1]);
    int H = out_sz[0];
    int W = out_sz[1];
    int C = out_sz[2];
    int N = out_sz[3];
    const mwSize out_size[] = {H, W, C, N};
    
    double *filter = mxGetPr(prhs[2]);
    int filter_h = filter[0];
    int filter_w = filter[1];
    int filter_sz = filter_h*filter_w;
    
    int stride = mxGetScalar(prhs[3]);
    
    int HH = (H - filter_h) / stride + 1;
    int WW = (W - filter_w) / stride + 1;
    int HHWW = HH * WW;
    
    const mwSize *sz = mxGetDimensions(M_IN);
    // input cols size: [C*filter_h*filter_w, N*HH*WW]
    int cols_H = sz[0];
    int cols_W = sz[1];
    
    // out size = [H, W, C, N]
    M_OUT = mxCreateNumericArray(4, out_size, mxDOUBLE_CLASS, mxREAL);
    B = mxGetPr(M_OUT);
    
    i = 0;  // i, j indicates starter position M_OUT.
    j = 0;   
    for(w = 0; w < WW; w++) {
        for(h = 0; h < HH; h++) {
            for(n = 0; n < N; n++) {
                int A_w = n + h*N + w*HH*N;
                
                for(c = 0; c < C; c++) {
                    for(ww = 0; ww < filter_w; ww++) {
                        for(hh = 0; hh < filter_h; hh++) {
                            int A_h = hh + ww*filter_h + c*filter_sz;
                            int t1 = i+hh;
                            int t2 = j+ww;
                            B(t1, t2, c, n) = B(t1, t2, c, n) + A(A_h, A_w);
                            //mexPrintf("%lf ", A(A_h, A_w));
                        }
                    }
                }
            }
            i = i + stride;
        }
        i = 0;
        j = j + stride;
    }
    
    
    
    
    // mexPrintf("%lf %lf", A(1,1,1,1), A(4,2,2,2));
    
    //M_OUT = mxCreateDoubleMatrix(C*filter_h*filter_w, N*HH*WW, mxREAL);
    //B = mxGetPr(M_OUT);     // ptr B point to M_OUT data
    
    
    
    return;
}