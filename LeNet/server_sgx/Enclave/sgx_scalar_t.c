#include "sgx_scalar_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


typedef struct ms_Scalar_SGX_t {
	uint64_t* ms__data;
} ms_Scalar_SGX_t;

typedef struct ms_Scalar_SGX_val_t {
	uint64_t* ms__data;
	uint64_t* ms_val;
} ms_Scalar_SGX_val_t;

typedef struct ms_Scalar_SGX_size_t {
	size_t* ms_size;
	uint64_t* ms__data;
} ms_Scalar_SGX_size_t;

typedef struct ms_Scalar_SGX_set_t {
	uint64_t* ms__data;
	uint64_t* ms_val;
} ms_Scalar_SGX_set_t;

static sgx_status_t SGX_CDECL sgx_Scalar_SGX(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Scalar_SGX_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Scalar_SGX_t* ms = SGX_CAST(ms_Scalar_SGX_t*, pms);
	ms_Scalar_SGX_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_Scalar_SGX_t), ms, sizeof(ms_Scalar_SGX_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp__data = __in_ms.ms__data;
	size_t _len__data = 1 * sizeof(uint64_t);
	uint64_t* _in__data = NULL;

	if (sizeof(*_tmp__data) != 0 &&
		1 > (SIZE_MAX / sizeof(*_tmp__data))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp__data, _len__data);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp__data != NULL && _len__data != 0) {
		if ( _len__data % sizeof(*_tmp__data) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in__data = (uint64_t*)malloc(_len__data)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in__data, 0, _len__data);
	}
	Scalar_SGX(_in__data);
	if (_in__data) {
		if (memcpy_verw_s(_tmp__data, _len__data, _in__data, _len__data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in__data) free(_in__data);
	return status;
}

static sgx_status_t SGX_CDECL sgx_Scalar_SGX_val(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Scalar_SGX_val_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Scalar_SGX_val_t* ms = SGX_CAST(ms_Scalar_SGX_val_t*, pms);
	ms_Scalar_SGX_val_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_Scalar_SGX_val_t), ms, sizeof(ms_Scalar_SGX_val_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp__data = __in_ms.ms__data;
	size_t _len__data = 1 * sizeof(uint64_t);
	uint64_t* _in__data = NULL;
	uint64_t* _tmp_val = __in_ms.ms_val;
	size_t _len_val = sizeof(uint64_t);
	uint64_t* _in_val = NULL;

	if (sizeof(*_tmp__data) != 0 &&
		1 > (SIZE_MAX / sizeof(*_tmp__data))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp__data, _len__data);
	CHECK_UNIQUE_POINTER(_tmp_val, _len_val);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp__data != NULL && _len__data != 0) {
		if ( _len__data % sizeof(*_tmp__data) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in__data = (uint64_t*)malloc(_len__data)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in__data, 0, _len__data);
	}
	if (_tmp_val != NULL && _len_val != 0) {
		if ( _len_val % sizeof(*_tmp_val) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_val = (uint64_t*)malloc(_len_val);
		if (_in_val == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_val, _len_val, _tmp_val, _len_val)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	Scalar_SGX_val(_in__data, _in_val);
	if (_in__data) {
		if (memcpy_verw_s(_tmp__data, _len__data, _in__data, _len__data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in__data) free(_in__data);
	if (_in_val) free(_in_val);
	return status;
}

static sgx_status_t SGX_CDECL sgx_Scalar_SGX_size(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Scalar_SGX_size_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Scalar_SGX_size_t* ms = SGX_CAST(ms_Scalar_SGX_size_t*, pms);
	ms_Scalar_SGX_size_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_Scalar_SGX_size_t), ms, sizeof(ms_Scalar_SGX_size_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	size_t* _tmp_size = __in_ms.ms_size;
	size_t _len_size = 1 * sizeof(size_t);
	size_t* _in_size = NULL;
	uint64_t* _tmp__data = __in_ms.ms__data;
	size_t _len__data = sizeof(uint64_t);
	uint64_t* _in__data = NULL;

	if (sizeof(*_tmp_size) != 0 &&
		1 > (SIZE_MAX / sizeof(*_tmp_size))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_size, _len_size);
	CHECK_UNIQUE_POINTER(_tmp__data, _len__data);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_size != NULL && _len_size != 0) {
		if ( _len_size % sizeof(*_tmp_size) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_size = (size_t*)malloc(_len_size)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_size, 0, _len_size);
	}
	if (_tmp__data != NULL && _len__data != 0) {
		if ( _len__data % sizeof(*_tmp__data) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in__data = (uint64_t*)malloc(_len__data);
		if (_in__data == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in__data, _len__data, _tmp__data, _len__data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	Scalar_SGX_size(_in_size, _in__data);
	if (_in_size) {
		if (memcpy_verw_s(_tmp_size, _len_size, _in_size, _len_size)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_size) free(_in_size);
	if (_in__data) free(_in__data);
	return status;
}

static sgx_status_t SGX_CDECL sgx_Scalar_SGX_set(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Scalar_SGX_set_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Scalar_SGX_set_t* ms = SGX_CAST(ms_Scalar_SGX_set_t*, pms);
	ms_Scalar_SGX_set_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_Scalar_SGX_set_t), ms, sizeof(ms_Scalar_SGX_set_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	uint64_t* _tmp__data = __in_ms.ms__data;
	size_t _len__data = sizeof(uint64_t);
	uint64_t* _in__data = NULL;
	uint64_t* _tmp_val = __in_ms.ms_val;
	size_t _len_val = 1 * sizeof(uint64_t);
	uint64_t* _in_val = NULL;

	if (sizeof(*_tmp_val) != 0 &&
		1 > (SIZE_MAX / sizeof(*_tmp_val))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp__data, _len__data);
	CHECK_UNIQUE_POINTER(_tmp_val, _len_val);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp__data != NULL && _len__data != 0) {
		if ( _len__data % sizeof(*_tmp__data) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in__data = (uint64_t*)malloc(_len__data)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in__data, 0, _len__data);
	}
	if (_tmp_val != NULL && _len_val != 0) {
		if ( _len_val % sizeof(*_tmp_val) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_val = (uint64_t*)malloc(_len_val);
		if (_in_val == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_val, _len_val, _tmp_val, _len_val)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	Scalar_SGX_set(_in__data, _in_val);
	if (_in__data) {
		if (memcpy_verw_s(_tmp__data, _len__data, _in__data, _len__data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in__data) free(_in__data);
	if (_in_val) free(_in_val);
	return status;
}

static sgx_status_t SGX_CDECL sgx_Scalar_SGX_test(void* pms)
{
	sgx_status_t status = SGX_SUCCESS;
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	Scalar_SGX_test();
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[5];
} g_ecall_table = {
	5,
	{
		{(void*)(uintptr_t)sgx_Scalar_SGX, 0, 0},
		{(void*)(uintptr_t)sgx_Scalar_SGX_val, 0, 0},
		{(void*)(uintptr_t)sgx_Scalar_SGX_size, 0, 0},
		{(void*)(uintptr_t)sgx_Scalar_SGX_set, 0, 0},
		{(void*)(uintptr_t)sgx_Scalar_SGX_test, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
} g_dyn_entry_table = {
	0,
};


