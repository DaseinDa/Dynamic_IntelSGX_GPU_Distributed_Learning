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
int main_vector_app();

#endif /* !_APP_H_ */
void vector_test();


template <std::unsigned_integral IntT>
class Vector{

    public:
        //using dev_type=SGX;
        using data_type = IntT;
        sgx_enclave_id_t enclave_eid=2;
        data_type *_data=nullptr;
        size_t _size=0;
        /// @brief 
        constexpr explicit Vector() noexcept{

            if(typeid(IntT)==typeid(uint64_t)){
                _data = (data_type *)malloc(sizeof(data_type));

                printf("##################The type of IntT is uint64 at Vector class ####################\n");

    
                Vector_SGX(enclave_eid,_data);//which it onto SGX

                _size =1;

                if(_data[0]==0){
                             printf("SGX interface for Vector_SGX invoke successful\n");
                      }else{
                             printf("SGX interface for Vector_SGX invoke failed\n");
                         }   
                }
        }

        ///
        constexpr explicit Vector(std::size_t n) noexcept{

             if(typeid(IntT)==typeid(uint64_t)){
                _size = n;
                _data = (uint64_t *)calloc(_size, sizeof(uint64_t));

                Vector_SGX_N(enclave_eid, _data,  _size);
        }
    }

        /// @brief copy il to _data
        constexpr Vector(std::initializer_list<data_type> il) noexcept{
             _size = il.size();
            _data = (data_type *)calloc(_size, sizeof(data_type));
            //memcpy(_data, il.begin(), sizeof(data_type)*_size);
            for(int i=0;i<_size;i++){
                _data[i]=il.begin()[i];
            }
    }

    ~Vector() {
        if (_data != nullptr) {
            free(_data);
        }
    }
    // Vector(const Vector& other) {
    //     _data = calloc(other._size, sizeof(data_type));
    //     memcpy(_data, other.data, _size * sizeof(data_type));
    //     return *this;
    // }
    Vector(Vector &&other) noexcept {
        _data = exchange(other._data, nullptr);
        enclave_eid = other.enclave_eid;
    }
    // auto operator=(const Vector& other) -> Vector& {
    //     if (this == &other) {
    //         return *this;
    //     }

    //     if(_data != nullptr) {
    //         free(_data);
    //     }
    //     _data = calloc(other._size, sizeof(data_type));
    //     memcpy(_data, other.data, _size * sizeof(data_type));
    //     return *this;
    // }
    auto operator=(Vector&& other) -> Vector& {
        using std::swap;
        swap(_data, other._data);
        enclave_eid = other.enclave_eid;
        return *this;
    }


    auto operator+=(const data_type rhs) -> Vector<data_type>& {

        Vector_SGX_Plus(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }

    
    auto operator-=(const data_type rhs) -> Vector<data_type>& {


        Vector_SGX_Minus(enclave_eid,_data,_data,rhs,_size);
 
        return *this;
    }

    
    auto operator*=(const data_type rhs) -> Vector<data_type>& {


        Vector_SGX_Mul(enclave_eid,_data,_data,rhs,_size);
 
        return *this;
    }



    
        
    auto operator/=(const data_type rhs) -> Vector<data_type>& {


        Vector_SGX_Divi(enclave_eid,_data,_data,rhs,_size);
 
        return *this;
    }

    auto operator&=(const data_type rhs) -> Vector<data_type>& {

        Vector_SGX_And(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }
    auto operator|=(const data_type rhs) -> Vector<data_type>& {
        
        Vector_SGX_OR(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }

    auto operator^=(const data_type rhs) -> Vector<data_type>& {
        Vector_SGX_XOR(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }

    auto operator<<=(const data_type rhs) -> Vector<data_type>& {
        Vector_SGX_Lshift(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }

    auto operator>>=(const data_type rhs) -> Vector<data_type>& {
        Vector_SGX_Rshift(enclave_eid,_data,_data,rhs,_size);
        return *this;
    }

/*Vector SGX interface vv*/
    ///@}

    /** @name Vector Operations
     *
     */
    ///@{
    /**
     *
     * ## Examples
     *
     * \snippet{trimleft} dev/gpu/vector_test.cu ops::vector
     */

    auto operator+=(const Vector<data_type>& rhs) -> Vector<data_type>& {

       Vector_SGX_Plus_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator-=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_Minus_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator*=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_Mul_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator/=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_Divi_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator&=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_And_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator|=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_OR_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator^=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_XOR_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator<<=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_Lshift_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }
    auto operator>>=(const Vector<data_type>& rhs) -> Vector<data_type>& {

        Vector_SGX_Rshift_vv(enclave_eid,_data,_data,rhs._data,_size);
        return *this;
    }

};

