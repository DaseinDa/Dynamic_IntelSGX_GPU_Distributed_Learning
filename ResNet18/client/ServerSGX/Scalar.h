/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#ifndef _APP_H_
#define _APP_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <typeinfo>
#include "sgx_error.h"       /* sgx_status_t */
#include "sgx_eid.h"     /* sgx_enclave_id_t */

//header files using byz hybrid platform global standard
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <iterator>
#include <iostream>
#include "Enclave_u.h"

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

# define ENCLAVE_FILENAME "enclave.signed.so"

extern sgx_enclave_id_t global_eid;    /* global enclave id */

#if defined(__cplusplus)
extern "C" {
#endif

void edger8r_array_attributes(void);
void edger8r_type_attributes(void);
void edger8r_pointer_attributes(void);
void edger8r_function_attributes(void);

void ecall_libc_functions(void);
void ecall_libcxx_functions(void);
void ecall_thread_functions(void);
#if defined(__cplusplus)
}
#endif
int main_app();

#endif /* !_APP_H_ */




template <std::unsigned_integral IntT>
class Scalar{
    public:
        //using device_type = SGX
       using data_type = IntT;
       sgx_enclave_id_t enclave_eid=0;
       //uint64_t *_data = (uint64_t *)malloc(sizeof(uint64_t));
       //data_type *_data = (data_type *)malloc(sizeof(data_type));
       data_type *_data=nullptr;

       //constexpr explicit Scalar() noexcept = default;
        constexpr Scalar() noexcept{
    
        if(typeid(IntT)==typeid(uint64_t)){

                printf("##################The type of IntT is uint64####################\n");

                printf("Here is constructor function Scalar scalar\n");
                 _data = (uint64_t *)calloc(1, sizeof(uint64_t));
                Scalar_SGX(enclave_eid,_data);
                printf("The enclave_eid at Scalar() Constructor is:%ld\n", enclave_eid);

                if(_data[0]==0){
                             printf("SGX interface for Scalar_SGX_ invoke successful\n");
                      }else{
                             printf("SGX interface for Scalar_SGX invoke failed\n");
                         }   
                }
}


       constexpr Scalar(data_type val) noexcept{

        printf("Here is constructor function Scalar scalar\n");
        /*Transfer the datatype to be specified before invoking SGX interface*/
        //uint64_t *_data = (uint64_t *)malloc(sizeof(uint64_t));

        if(typeid(IntT)==typeid(uint64_t)){

            printf("##################The type of IntT is uint64####################\n");
            uint64_t  value= val;
            _data = (uint64_t *)calloc(1, sizeof(uint64_t));
            
            printf("The enclave_eid at Scalar() Constructor is:%ld\n", enclave_eid);
            Scalar_SGX_val(enclave_eid,_data,&value);
                    
            if(_data[0]==val){
                    printf("SGX interface for Scalar_SGX_val invoke successful\n");
                }else{
                    printf("SGX interface for Scalar_SGX_val invoke failed\n");
                }       
            }
}


        ~Scalar() = default;//Destructor execute when the life cycle of class end

         Scalar(const Scalar& other) = default;

        auto operator=(const Scalar& other) -> Scalar& = default;

    /// Get the iterator to the beginning of the Scalar.

        auto begin() { return _data; }
        const auto begin() const { return *_data; }
    /// Get the iterator to the end of the Scalar.
        auto end() { return *_data+0; }
        const auto end() const { return *_data+0; }


    /// Get the pointer to raw data.

        auto data() { return *_data; }

        const auto data() const { return *_data; }


        size_t size() const{
            printf("Here is Scalar::size()!\n");
            size_t size = 0;
            if(typeid(IntT)==typeid(uint64_t)){
                printf("##################The type of IntT is uint64####################\n");
                uint64_t *_data_sgx = (uint64_t *)malloc(sizeof(uint64_t));
                memcpy(_data_sgx, _data,sizeof(_data));
                Scalar_SGX_size(enclave_eid, &size, _data_sgx);
            }
            return size;
        }

        void get()=delete;

        void set(data_type val){
            if(typeid(IntT)==typeid(uint64_t)){
                    printf("##################The type of IntT is uint64####################\n");
                    uint64_t  value= val;
                    uint64_t *_data_sgx = (uint64_t *)malloc(sizeof(uint64_t));
                    Scalar_SGX_set(enclave_eid, _data_sgx, &value);
                    memcpy(_data, _data_sgx,sizeof(_data_sgx));         
                }
        }
        
        
        // void set(data_type val);

        // auto get() -> data_type;



    
};