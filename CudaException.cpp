#include "CudaException.h"

CudaException::CudaException(cudaError_t error, uint32_t line)
	: m_error(error), m_line(line)
{
}

char const* CudaException::what() const
{
	return cudaGetErrorString(m_error);
}

void CudaException::throwIfFailed(cudaError_t error, uint32_t line)
{
	if (error != cudaSuccess)
		throw CudaException(error, line);
}
