#include <stdexcept>
#include <device_launch_parameters.h>
#include "Matrix.cuh"
#include "CudaException.h"
#include "CudaPtr.h"

void matrixMultiplication(float const* a, float const* b, float* c, size_t l, size_t m, size_t n, Matrix::MulMode mode);

const size_t WARP_SIZE = 32;

void printa(float* a, size_t l, size_t h)
{
	for (size_t i = 0; i < l; i++) {
		for (size_t j = 0; j < h; j++)
			printf("%.0f ", a[i * h + j]);
		printf("\n");
	}
}

Matrix::Matrix(size_t h, size_t w)
	: m_data(nullptr), m_h(h), m_w(w)
{
	m_data = new float[h * w];
}

Matrix::Matrix(Matrix const& other) : Matrix(other.m_h, other.m_w)
{
	memcpy(m_data, other.m_data, m_h * m_w * sizeof(float));
}

Matrix& Matrix::operator=(Matrix const& other)
{
	delete m_data;
	m_h = other.m_h;
	m_w = other.m_w;
	m_data = new float[m_h * m_w];
	memcpy(m_data, other.m_data, m_h * m_w * sizeof(float));

	return *this;
}

Matrix Matrix::full(float val, size_t h, size_t w)
{
	Matrix fullMatrix(h, w);
	for (size_t i = 0; i < h * w; i++)
		fullMatrix.m_data[i] = val;

	return fullMatrix;
}

Matrix Matrix::mul(Matrix const& other, MulMode mode) const
{
	if (this->m_w != other.m_h)
		throw std::invalid_argument("Matrix dimension mismatch");

	Matrix productMatrix(this->m_h, other.m_w);
	
	matrixMultiplication(this->m_data, other.m_data, productMatrix.m_data, this->m_h, this->m_w, other.m_w, mode);

	return productMatrix;
}

size_t Matrix::hight() const
{
	return m_h;
}

size_t Matrix::width() const
{
	return m_w;
}

float Matrix::at(size_t i, size_t j) const
{
	return m_data[i * m_w + j];
}

Matrix::~Matrix()
{
	delete m_data;
	m_data = nullptr;
}

//-----------------------CUDA---------------------------//

__global__ void simpleMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

	if (cRow >= l || cCol >= n)
		return;

	float sum = 0;
	for (size_t i = 0; i < m; i++)
		sum += a[cRow * m + i] * b[i * n + cCol];
	c[cRow * n + cCol] = sum;
}

__global__ void sharedMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
	size_t tileCol = threadIdx.x;
	size_t tileRow = threadIdx.y;

	__shared__ float aTile[WARP_SIZE][WARP_SIZE];
	__shared__ float bTile[WARP_SIZE][WARP_SIZE + 1]; // + 1 to avoid bank conflicts
	
	float cVal = 0.f;
	bool isOutOfC = cRow >= l || cCol >= n;

	for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
	{	
		aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * WARP_SIZE + tileCol)] : 0.f;
		bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * WARP_SIZE + tileRow) * n + cCol] : 0.f;
		__syncthreads();
		
		for (size_t i = 0; i < WARP_SIZE; i++)
			cVal += aTile[tileRow][i] * bTile[i][tileCol];
		__syncthreads();
	}
	if (!isOutOfC)
		c[cRow * n + cCol] = cVal;
}

__global__ void warpIntrinsicsMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
	size_t tileCol = threadIdx.x;
	size_t tileRow = threadIdx.y;

	__shared__ float aTile[WARP_SIZE][WARP_SIZE];
	__shared__ float bTile[WARP_SIZE][WARP_SIZE + 1]; // + 1 to avoid bank conflicts

	float cVal = 0.f;
	bool isOutOfC = cRow >= l || cCol >= n;

	for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
	{
		aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * WARP_SIZE + tileCol)] : 0.f;
		bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * WARP_SIZE + tileRow) * n + cCol] : 0.f;
		__syncthreads();

		float aTileLocal = aTile[tileRow][tileCol];
		for (size_t i = 0; i < WARP_SIZE; i++)
			cVal += __shfl_sync(0xffffffff, aTileLocal, i) * bTile[i][tileCol];
		__syncthreads();
	}
	if (!isOutOfC)
		c[cRow * n + cCol] = cVal;
}

void matrixMultiplication(float const* a, float const* b, float* c, size_t l, size_t m, size_t n, Matrix::MulMode mode)
{
	CUDA_FAIL(cudaSetDevice(0));
	
	CudaUniquePtr<float> aDev(nullptr), bDev(nullptr), cDev(nullptr);

	CUDA_FAIL(cudaMalloc(&aDev, l * m * sizeof(float)));
	CUDA_FAIL(cudaMalloc(&bDev, m * n * sizeof(float)));
	CUDA_FAIL(cudaMalloc(&cDev, l * n * sizeof(float)));

	CUDA_FAIL(cudaMemcpy(aDev.get(), a, l * m * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_FAIL(cudaMemcpy(bDev.get(), b, m * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockInGrid((n-1ULL) / WARP_SIZE + 1ULL, (l-1ULL) / WARP_SIZE + 1ULL);
	dim3 threadInBlock(WARP_SIZE, WARP_SIZE);
	switch (mode) {
	case Matrix::MulMode::SIMPLE:
		simpleMatMul <<< blockInGrid, threadInBlock >>> (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	case Matrix::MulMode::SHARED:
		sharedMatMul <<< blockInGrid, threadInBlock >>> (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	case Matrix::MulMode::INTRINSICS:
		warpIntrinsicsMatMul << < blockInGrid, threadInBlock >> > (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	}

	CUDA_FAIL(cudaDeviceSynchronize());
	CUDA_FAIL(cudaMemcpy(c, cDev.get(), l * n * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_FAIL(cudaGetLastError());
	//printa(c, l, n);
}


