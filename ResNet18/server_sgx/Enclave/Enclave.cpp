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

#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <cstdint>
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include "shim.hpp"
#include <Eigen/Dense>
#include <omp.h>
using namespace Eigen;
#define FOREACH(i,count) for (int i = 0; i < count; ++i)
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}


void sgx_decode(uint64_t *_data){

	_data[0]=1;

	/*The decode body decode ciphertext from REE*/

}

void sgx_encode(uint64_t *_data){
	/*The encode body used return to REE*/
}


/*Scalar SGX*/

void Scalar_SGX(uint64_t *_data){


	/*The SGX indicated operation*/
	// typedef Matrix<uint64_t,1,0> SGX_Scalar;
	// SGX_Scalar test = SGX_Scalar::Zero(1);
	_data[0]=0;
	//data.data();
	printf("Here is Scalar_SGX for _data, Constructor for Scalar()\n");
	/*The encode body to return encrypted text to REE*/
}


void Scalar_SGX_val(uint64_t *_data, uint64_t * val){



	printf("###########################Scalar_SGX_val ######################################################\n");
	memcpy(_data, val, sizeof(val));
	printf("###########################Scalar_SGX_val ######################################################\n");
	printf("Here is Scalar_SGX for _data, Constructor for Scalar()\n");
	/*The encode body to return encrypted text to REE*/
}
void Scalar_SGX_size(size_t * size, uint64_t * _data){

	size[0] = sizeof(_data)/sizeof(typeid(_data).name());
	printf("The size of Scalar at SGX is: %ld\n",size[0]);
	printf("###########################Scalar_SGX_size ######################################################\n");
}
void Scalar_SGX_set(uint64_t * _data,  uint64_t * val){

	_data[0] = val[0];
}
void Scalar_SGX_test(){
	printf("######################Here is test function testing edl interface usage ability around REE\n");
}


void relu_sgx(double * data_out, double * data_in, size_t n){
	printf("Here is relu_sgx\n");
	start_timer();
#pragma omp parallel for num_threads(9)
	for(int i =0;i<n;i++){
		data_out[i] = data_in[i]*(data_in[i]>0);
		// printf("thread:%d, iteration:%d\n",omp_get_thread_num(),i);
		// printf("total threads:%d\n",omp_get_num_threads());
	}
	stop_timer();
}

void maxpool_2_sgx(double * data_out, double * data_in, int* shape_sgx, size_t n_out, size_t n_in){
	printf("Here is maxpool kernel size of 2 hhhh\n");
	double data[shape_sgx[0]][shape_sgx[1]][shape_sgx[2]][shape_sgx[3]];
	double data_max[shape_sgx[0]][shape_sgx[1]][shape_sgx[2]/2][shape_sgx[3]/2];
	// printf("The n_in is%d\n",n_in);
	// for(int i=0;i<n_in;i++){
    //     printf("The enclave maxpool input is:%f\n",data_in[i]);
    // }
	/*转换成四维数组*/
	memcpy(data,data_in,n_in*sizeof(data_in[0]));
	// start_timer();
#pragma omp parallel for
	for(int i = 0;i<shape_sgx[0];i++){
		for(int j=0;j<shape_sgx[1];j++){
			for(int o0=0;o0<shape_sgx[2]/2;o0++){
				for(int o1=0;o1<shape_sgx[3]/2;o1++){
					int x0 = 0, x1 = 0, ismax;	
					for(int l0=0;l0<2;l0++){
					for(int l1=0;l1<2;l1++){
						ismax = data[i][j][o0*2+l0][o1*2+l1]>data[i][j][o0*2+x0][o1*2+x1];
						x0 +=ismax*(l0-x0);
						x1 +=ismax*(l1-x1);
					}
					}
				data_max[i][j][o0][o1]=data[i][j][o0*2+x0][o1*2+x1];
				}
			}
		}
	}
	memcpy(data_out,data_max,n_out*sizeof(data_out[0]));
	// printf("The maxpool in sgx execute successfully\n");
	// for(int i=0;i<n_out;i++){
    //     if(data_out[i]!=0){
    //     printf("The enclave maxpool output is:%f\n",data_out[i]);
    //     }
    // }
	
	// stop_timer();

}

void grad_relu_sgx(double * data_out, double * data_in, size_t n){
	printf("Here is grad_relu_sgx\n");
	start_timer();
#pragma omp parallel for num_threads(9)
	for(int i =0;i<n;i++){
		data_out[i] = data_in[i]*(data_in[i]>0);
		// printf("thread:%d, iteration:%d\n",omp_get_thread_num(),i);
		// printf("total threads:%d\n",omp_get_num_threads());
	}
	stop_timer();
}

void grad_maxpool_2_sgx(double * data_out_error, double * data_in_error, double * data_input,int* shape_input, size_t n_out, size_t n_in){
	printf("Here is grad maxpool kernel size of 2 hhhh\n");
	double input[shape_input[0]][shape_input[1]][shape_input[2]][shape_input[3]];
	double inerror[shape_input[0]][shape_input[1]][shape_input[2]][shape_input[3]];
	size_t max_size = (shape_input[2])/2;
	double outerror[shape_input[0]][shape_input[1]][max_size][max_size];
	memcpy(input,data_input,n_in*sizeof(data_input[0]));
	memcpy(outerror,outerror,n_out*sizeof(data_input[0]));
	int len0=2;
	int len1=2;
#pragma omp parallel for
	FOREACH(b, GETLENGTH(outerror))		
	FOREACH(i, GETLENGTH(*outerror))																\
	FOREACH(o0, GETLENGTH(**(outerror)))															\
	FOREACH(o1, GETLENGTH(***(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[b][i][o0*len0 + l0][o1*len1 + l1] > input[b][i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[b][i][o0*len0 + x0][o1*len1 + x1] = outerror[b][i][o0][o1];						\
		// printf("grad maxpool test: %i\n",b);																		
	}																							\
	memcpy(data_in_error,inerror,n_in*sizeof(double));
}

void maxpool_str_sgx(double* data_out,double *data_in,int* shape_sgx, size_t n_out, size_t n_in,size_t kernel_size, size_t stride){

	double input[shape_sgx[0]][shape_sgx[1]][shape_sgx[2]][shape_sgx[3]];
	size_t output_length = ((shape_sgx[2]-kernel_size)/stride) +1;
	double output[shape_sgx[0]][shape_sgx[1]][output_length][output_length];
	memcpy(input,data_in,n_in*sizeof(double));
	int len0, len1=kernel_size;
	int s =stride;
	FOREACH(b, GETLENGTH(output))		
	FOREACH(i,GETLENGTH(*output))																\
		FOREACH(o0,GETLENGTH(**(output)))														\
			FOREACH(o1,GETLENGTH(***(output)))													\
				{																				\
					int x0=0, x1=0, ismax;														\
					FOREACH(l0,len0)															\
						FOREACH(l1,len1)														\
						{																		\
							ismax = input[b][i][o0*s + l0][o1*s + l1] > input[b][i][o0*s+ x0][o1*s + x1];\
							x0 += ismax * (l0 - x0);											\
							x1 += ismax * (l1 - x1);											\
						}																		\
					output[b][i][o0][o1] = input[b][i][o0*s + x0][o1*s + x1];							\
				}																				\
	memcpy(data_out,output,n_out*sizeof(double));
	}																							\
	


void grad_maxpool_str_sgx(double * data_out_error, double * data_in_error, double * data_input,int* shape_input, size_t n_out, size_t n_in, size_t kernel,size_t stride){
	printf("Here is grad maxpool kernel size of 2 hhhh\n");
	double input[shape_input[0]][shape_input[1]][shape_input[2]][shape_input[3]];
	double inerror[shape_input[0]][shape_input[1]][shape_input[2]][shape_input[3]];
	size_t max_size = (shape_input[2]-kernel)/stride-1;
	double outerror[shape_input[0]][shape_input[1]][max_size][max_size];
	memcpy(input,data_input,n_in*sizeof(data_input[0]));
	memcpy(outerror,outerror,n_out*sizeof(data_input[0]));
	const int len0 = kernel;																\
	const int len1 = kernel;																\
	const int s = stride;	
	FOREACH(b, GETLENGTH(outerror))															\
	FOREACH(i, GETLENGTH(*outerror))																\
	FOREACH(o0, GETLENGTH(**(outerror)))															\
	FOREACH(o1, GETLENGTH(***(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[b][i][o0*s + l0][o1*s + l1] > input[b][i][o0*s + x0][o1*s + x1];		\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[b][i][o0*s + x0][o1*s + x1]=outerror[b][i][o0][o1];								\
	}																	\
	memcpy(data_in_error,inerror,n_in*sizeof(double));
}

void test(){
	printf("Here is SGX Enclave test\n");

}
