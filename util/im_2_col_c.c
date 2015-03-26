#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    #define M_IN prhs[0]
    #define M_OUT plhs[0]
    //#define A(i,j) A[i+j*H]  // starts from 0
    #define A(h, w, c, n) A[h + w*H + c*H*W + n*C*H*W]
    
    double *A = mxGetPr(M_IN);  // A point to X_padded of size [H,W,C,N]
    double *B;      // M_OUT data ptr
    
    double *filter = mxGetPr(prhs[1]);
    int filter_h = filter[0];
    int filter_w = filter[1];
    int filter_sz = filter_h*filter_w;
    
    // loop variables
    int h, w, c, n; // for [H, W, C, N]
    int ww, hh; // for [filter_w, filter_h]
    
    int stride = mxGetScalar(prhs[2]);
    
    // int d = mxGetNumberOfDimensions(M_IN);
     
    const mwSize *sz = mxGetDimensions(M_IN);
    int H = sz[0];
    int W = sz[1];
    int C = sz[2];
    int N = sz[3];
    
    int HH = (H-filter_h)/stride + 1;
    int WW = (W-filter_w)/stride + 1;
    
    // mexPrintf("%lf %lf", A(1,1,1,1), A(4,2,2,2));
    
    M_OUT = mxCreateDoubleMatrix(C*filter_h*filter_w, N*HH*WW, mxREAL);
    
    B = mxGetPr(M_OUT);     // ptr B point to M_OUT data
    
    
    int x, y, B_h, B_w, t1, t2;
    for(w = 0; w < WW; w++) {
        for(h = 0; h < HH; h++) {
            x = w*stride;
            y = h*stride;
            
            for(n = 0; n < N; n++) {
                for(c = 0; c < C; c++) {
                    for(ww = 0; ww < filter_w; ww++) {
                        for(hh = 0; hh < filter_h; hh++) {
                            B_h = hh + ww*filter_h + c*filter_sz;
                            B_w = n + h*N + w*N*HH;
                            t1 = y+hh;  // use tmp variable to match 'define' format
                            t2 = x+ww;
                            B[B_h + B_w*C*filter_h*filter_w] = A(t1, t2, c, n);
                          
                            // B[hh + ww*filter_h + c*filter_sz, n + h*N + w*N*HH] = A(y + hh, w + ww, c, n);
                            // mexPrintf("%lf ", A(y + hh,x + ww,c,n));
                            // mexPrintf("%d %d %d %d  ", y + hh, x + ww, c, n);
                            // mexPrintf("%lf ", A(y + hh, x + ww, c, n));
                        }
                        // mexPrintf("\n");
                    }
                }
            }
        }
    }
    
    return;
}













