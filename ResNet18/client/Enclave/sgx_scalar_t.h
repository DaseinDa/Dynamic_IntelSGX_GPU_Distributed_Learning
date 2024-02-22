#ifndef SGX_SCALAR_T_H__
#define SGX_SCALAR_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void Scalar_SGX(uint64_t* _data);
void Scalar_SGX_val(uint64_t* _data, uint64_t* val);
void Scalar_SGX_size(size_t* size, uint64_t* _data);
void Scalar_SGX_set(uint64_t* _data, uint64_t* val);
void Scalar_SGX_test(void);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
