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

#endif /* !_APP_H_ */


template <std::unsigned_integral IntT>
class Matrix{

    public:
        //using dev_type=SGX;
        using data_type = IntT;
        sgx_enclave_id_t enclave_eid=2;
        data_type *_data=nullptr;
        size_t _height=0;
        size_t _width=0;
        /// @brief 
  
        constexpr explicit Matrix() noexcept {
            if(typeid(IntT)==typeid(uint64_t)){
                _data = (data_type *)malloc(sizeof(data_type));
                //Matrix_SGX(enclave_eid,_data);
                _height=1;
                _width=1;

            }
        }

        constexpr explicit Matrix(std::size_t h, std::size_t w) noexcept{
                _height=h;
                _width=w;
                _data = (uint64_t *)calloc(_height*_width, sizeof(uint64_t));
                Matrix_SGX_H_W(enclave_eid, _data,_height,_height*_width);

        }

        constexpr Matrix(std::initializer_list<data_type> il, std::size_t h, std::size_t w) noexcept{
                _height=h;
                _width=w;
                _data = (uint64_t *)calloc(_height*_width, sizeof(uint64_t));
                memcpy(_data, il.begin(), sizeof(data_type)*il.size());

        }

            ~Matrix() = default;
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = delete;
    auto operator=(const Matrix& other) -> Matrix& = default;
    auto operator=(Matrix&& other) noexcept -> Matrix& = delete;
    data_type* data(){return _data;}
    auto height() const { return _height; }

    /// \copydoc hppcp::dev::Matrix::width
    auto width() const { return _width; }

    auto size() const { return _height*_width; }

    void reshape(std::size_t h, std::size_t w) {
        _height = h;
        _width = w;
    }

    void zero() { 

        Matrix_SGX_Zero(enclave_eid, _data,_height,_height*_width);

    }

    void one() {
        
        Matrix_SGX_One(enclave_eid, _data,_height,_height*_width);
    }

    void fill(data_type val) {

        Matrix_SGX_Fill(enclave_eid,_data,val,_height,_height*_width);
        
    }

    auto operator+=(const data_type rhs) -> Matrix<data_type>& {

        Matrix_SGX_Plus(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator-=(const data_type rhs) -> Matrix<data_type>& {
        Matrix_SGX_Minus(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator*=(const data_type rhs) -> Matrix<data_type>& {
        Matrix_SGX_Mul(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator/=(const data_type rhs) -> Matrix< data_type>& {
        Matrix_SGX_Divi(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }
    auto operator&=(const data_type rhs) -> Matrix< data_type>& {
        Matrix_SGX_And(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator|=(const data_type rhs) -> Matrix<data_type>& {
        Matrix_SGX_OR(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator^=(const data_type rhs) -> Matrix< data_type>& {
        Matrix_SGX_XOR(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator<<=(const data_type rhs) -> Matrix< data_type>& {
        Matrix_SGX_Lshift(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

    auto operator>>=(const data_type rhs) -> Matrix< data_type>& {
        Matrix_SGX_Rshift(enclave_eid,_data,_data,rhs,_height,_height*_width);
        return *this;
    }

/*Matrix SGX interface mv*/
    auto operator+=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Plus_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }

    auto operator-=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Minus_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }

    auto operator*=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Mul_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator/=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Divi_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator&=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_And_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator|=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_OR_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator^=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_XOR_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator<<=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Lshift_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator>>=(const Vector<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Rshift_mv(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    /*Matrix SGX interface mm*/

    auto operator+=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Plus_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator-=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Minus_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator*=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Mul_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator/=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Divi_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator&=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_And_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator|=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_OR_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator^=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_XOR_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator<<=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Lshift_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    auto operator>>=(const Matrix<data_type>& rhs) -> Matrix<data_type>& {
        Matrix_SGX_Rshift_mm(enclave_eid,_data,_data,rhs._data,_height, _height*_width);
        return *this;
    }
    void gemv(const Matrix<data_type>& x, Vector<data_type>& y) const {
        _height = x._height;
        _width =1;//x._width == y._height
        _data = (uint64_t *)calloc(_height, sizeof(uint64_t));
        Matrix_SGX_gemv(enclave_eid,data(), x._data, y._data, x._height, x._height*y._height);
    }

    void gemm(const Matrix<data_type>& x, Vector<data_type>& y) const {
    _height = x._height;
    _width =y._width;
    _data = (uint64_t *)calloc(_height, sizeof(uint64_t));//x._width = y._height
    Matrix_SGX_gemm(enclave_eid,data(), x._data, y._data, x._height*x._width, y._width*y._height, x._height, x._width, x._height*y._width);
}


};
