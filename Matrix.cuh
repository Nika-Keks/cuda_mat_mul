#pragma once
#include <cstddef>
#include <cstdint>


class Matrix {
public:
	enum MulMode
	{
		SIMPLE,
		SHARED,
		INTRINSICS
	};

	Matrix(size_t h, size_t w);
	Matrix(Matrix const& other);
	Matrix& operator=(Matrix const& other);
	static Matrix full(float val, size_t h, size_t w);
	static Matrix rand(size_t h, size_t w);
	Matrix mul(Matrix const& other, MulMode mode) const;
	float at(size_t i, size_t j) const;
	float& at(size_t i, size_t j);
	size_t hight() const;
	size_t width() const;

	virtual ~Matrix();

private:

	float* m_data;
	size_t m_h, m_w;
};