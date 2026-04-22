package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/opencl/headers -DCL_TARGET_OPENCL_VERSION=200 -Wno-deprecated-declarations
#cgo LDFLAGS: -L${SRCDIR} -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// kernel de multiplicação de matrizes
static const char* matmul_kernel =
"__kernel void matmul("
"   __global const float* A,"
"   __global const float* B,"
"   __global float* C,"
"   const int M, const int N, const int K)"
"{"
"   int row = get_global_id(0);"
"   int col = get_global_id(1);"
"   if (row < M && col < N) {"
"       float sum = 0.0f;"
"       for (int k = 0; k < K; k++) {"
"           sum += A[row * K + k] * B[k * N + col];"
"       }"
"       C[row * N + col] = sum;"
"   }"
"}";

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    int ready;
    char device_name[256];
    unsigned long mem_size;
} GPUContext;

static GPUContext gpu = {0};

int gpu_init() {
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return -1;

    // tenta GPU primeiro
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu.device, NULL);
    if (err != CL_SUCCESS) {
        // fallback pra CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &gpu.device, NULL);
        if (err != CL_SUCCESS) return -2;
    }

    clGetDeviceInfo(gpu.device, CL_DEVICE_NAME, 256, gpu.device_name, NULL);
    clGetDeviceInfo(gpu.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(unsigned long), &gpu.mem_size, NULL);

    gpu.context = clCreateContext(NULL, 1, &gpu.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return -3;

    gpu.queue = clCreateCommandQueue(gpu.context, gpu.device, 0, &err);
    if (err != CL_SUCCESS) return -4;

    // compila kernel
    gpu.program = clCreateProgramWithSource(gpu.context, 1, &matmul_kernel, NULL, &err);
    if (err != CL_SUCCESS) return -5;

    err = clBuildProgram(gpu.program, 1, &gpu.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) return -6;

    gpu.kernel = clCreateKernel(gpu.program, "matmul", &err);
    if (err != CL_SUCCESS) return -7;

    gpu.ready = 1;
    return 0;
}

const char* gpu_device_name() { return gpu.device_name; }
unsigned long gpu_mem_size() { return gpu.mem_size; }
int gpu_is_ready() { return gpu.ready; }

// multiplica A[M x K] * B[K x N] = C[M x N] na GPU
int gpu_matmul(float* A, float* B, float* C, int M, int K, int N) {
    if (!gpu.ready) return -1;

    cl_int err;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cl_mem bufA = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, A, &err);
    cl_mem bufB = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, B, &err);
    cl_mem bufC = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY, sizeC, NULL, &err);

    clSetKernelArg(gpu.kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(gpu.kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(gpu.kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(gpu.kernel, 3, sizeof(int), &M);
    clSetKernelArg(gpu.kernel, 4, sizeof(int), &N);
    clSetKernelArg(gpu.kernel, 5, sizeof(int), &K);

    size_t global[2] = {M, N};
    err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        return -2;
    }

    clEnqueueReadBuffer(gpu.queue, bufC, CL_TRUE, 0, sizeC, C, 0, NULL, NULL);
    clFinish(gpu.queue);

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    return 0;
}

void gpu_cleanup() {
    if (gpu.kernel) clReleaseKernel(gpu.kernel);
    if (gpu.program) clReleaseProgram(gpu.program);
    if (gpu.queue) clReleaseCommandQueue(gpu.queue);
    if (gpu.context) clReleaseContext(gpu.context);
    gpu.ready = 0;
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/azzidev/zendia.ai/pkg/matrix"
)

var Available bool

func Init() error {
	ret := C.gpu_init()
	if ret != 0 {
		return fmt.Errorf("GPU init falhou (código: %d)", ret)
	}
	Available = true
	name := C.GoString(C.gpu_device_name())
	mem := uint64(C.gpu_mem_size()) / 1024 / 1024
	fmt.Printf("🎮 GPU: %s (%d MB)\n", name, mem)
	return nil
}

func IsReady() bool {
	return Available && C.gpu_is_ready() == 1
}

// MatMul multiplica duas matrizes na GPU
// A[M x K] * B[K x N] = C[M x N]
func MatMul(a, b *matrix.Matrix) (*matrix.Matrix, error) {
	if !IsReady() {
		return a.Mul(b)
	}

	if a.Cols != b.Rows {
		return nil, fmt.Errorf("dimensões incompatíveis: (%d,%d) * (%d,%d)", a.Rows, a.Cols, b.Rows, b.Cols)
	}

	M, K, N := a.Rows, a.Cols, b.Cols

	// pra matrizes pequenas, CPU paralela é mais rápido (overhead de transferência)
	if M*K*N < 512 {
		return a.MulParallel(b)
	}

	// flatten pra float32 (GPU trabalha com float32)
	flatA := flatten64to32(a)
	flatB := flatten64to32(b)
	flatC := make([]float32, M*N)

	ret := C.gpu_matmul(
		(*C.float)(unsafe.Pointer(&flatA[0])),
		(*C.float)(unsafe.Pointer(&flatB[0])),
		(*C.float)(unsafe.Pointer(&flatC[0])),
		C.int(M), C.int(K), C.int(N),
	)

	if ret != 0 {
		// fallback pra CPU
		return a.Mul(b)
	}

	// converte de volta pra float64
	result := matrix.New(M, N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			result.Data[i][j] = float64(flatC[i*N+j])
		}
	}
	return result, nil
}

func Cleanup() {
	C.gpu_cleanup()
	Available = false
}

func flatten64to32(m *matrix.Matrix) []float32 {
	flat := make([]float32, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			flat[i*m.Cols+j] = float32(m.Data[i][j])
		}
	}
	return flat
}
