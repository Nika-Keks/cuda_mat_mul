#pragma once
#include <cuda_runtime.h>

template <typename T>
class CudaUniquePtr
{
public:
	CudaUniquePtr(T* ptr) : m_ptr(nullptr) {}

	CudaUniquePtr(CudaUniquePtr<T> const& other) = delete;
	CudaUniquePtr<T>& operator=(CudaUniquePtr<T> const& other) = delete;
	
	T** operator& ()
	{ 
		cudaFree(m_ptr);
		return &m_ptr; 
	}

	T const* get() const { return m_ptr; }
	T* get() { return m_ptr; }
	void reset()
	{
		cudaFree(m_ptr);
		m_ptr = nullptr;
	}
	~CudaUniquePtr() { cudaFree(m_ptr); }
private:
	T* m_ptr;
};