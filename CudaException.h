#pragma once
#include <cstdint>
#include <exception>
#include <cuda_runtime.h>

class CudaException : public std::exception
{
public:
	CudaException(cudaError_t error, uint32_t line);
	char const* what() const override;

	static void throwIfFailed(cudaError_t error, uint32_t line);

private:
	cudaError_t m_error;
	uint32_t m_line;
};

#define CUDA_FAIL(error) CudaException::throwIfFailed(error, __LINE__)