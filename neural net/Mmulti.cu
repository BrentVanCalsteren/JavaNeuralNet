
// Define the kernel function
extern "C"
__global__ void multiplication(float *A, float* B, float *C, int M, int N, int P) {
  // Compute the row and column indices for the current thread
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  int col = blockIdx.y*blockDim.y+threadIdx.y;

  // Check if the current thread is within the bounds of the matrices
  if (row < M && col < P) {
    float tmp_sum = 0.0f;
    // Perform the matrix multiplication
    for (int k = 0; k < N; k++) {
      tmp_sum += A[row * N + k] * B[k * P + col];
       if(tmp_sum > 1e8){
      tmp_sum = tmp_sum /1e6;
    } 
    else if(tmp_sum < -1e8){
       tmp_sum = tmp_sum /1e6;
    }
    else if(fabsf(tmp_sum) < 1e-8){
       tmp_sum =  tmp_sum*1e-6; 
    }  
    }
    C[row * P + col] = tmp_sum;
   
  }
}

extern "C"
__global__ void sum(float* A, float* B, float* C, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
    C[x] = A[x] + B[x];
  }
}

extern "C"
__global__ void subtract(float* A, float* B, float* C, int n)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
    C[x] = A[x] - B[x];
  }
}

extern "C"
__global__ void Hadamard(float* a, float* b, float* c, int n)
{
    // Compute the index of the element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within bounds
    if (i < n)
    {
        // Perform the operation and store the result in the output array
        c[i] = a[i] * b[i];
    }
}

extern "C"
__global__ void scale(float* in,float* out, float value, int length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    out[i] = in[i] * value;
  }
}

extern "C"
__global__ void transpose(float* in, float* out, int xsize, int ysize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < xsize && y < ysize) {
            out[x + y*xsize] = in[x*ysize + y];
    }
}

