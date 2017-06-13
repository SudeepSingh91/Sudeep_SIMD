#include <xmmintrin.h>

#define SHUFFLE(x, y, z, w) \
((x) | (y << 2) | (z << 4) | (w << 6))

#define _mm_replicate_x_ps(v) \
_mm_shuffle_ps((v), (v), SHUFFLE(0, 0, 0, 0))

#define _mm_replicate_y_ps(v) \
_mm_shuffle_ps((v), (v), SHUFFLE(1, 1, 1, 1))

#define _mm_replicate_z_ps(v) \
_mm_shuffle_ps((v), (v), SHUFFLE(2, 2, 2, 2))

#define _mm_replicate_w_ps(v) \
_mm_shuffle_ps((v), (v), SHUFFLE(3, 3, 3, 3))

#define _mm_madd_ps(a, b, c) \
_mm_add_ps(_mm_mul_ps(a, b), c)

__m128 VecMatMul(const __m128& i_vec, const __m128 i_mat[4])
{
	const __m128 x = _mm_replicate_x_ps(i_vec);
	const __m128 y = _mm_replicate_y_ps(i_vec);
	const __m128 z = _mm_replicate_z_ps(i_vec);
	const __m128 w = _mm_replicate_w_ps(i_vec);

	__m128 result = _mm_mul_ps(x, i_mat[0]);
	result = _mm_madd_ps(y, i_mat[1], result);
	result = _mm_madd_ps(z, i_mat[2], result);
	result = _mm_madd_ps(w, i_mat[3], result);

	return result;
}

void main()
{
	_declspec(align(16)) const float v[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
	__m128 row[4];

	row[0] = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
	row[1] = _mm_set_ps(2.0f, 2.0f, 2.0f, 2.0f);
	row[2] = _mm_set_ps(3.0f, 3.0f, 3.0f, 3.0f);
	row[3] = _mm_set_ps(4.0f, 4.0f, 4.0f, 4.0f);

	const __m128 vec = _mm_load_ps(&v[0]);

	const __m128 result = VecMatMul(vec, row);
}