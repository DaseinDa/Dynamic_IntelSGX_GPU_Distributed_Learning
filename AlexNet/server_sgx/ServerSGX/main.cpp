#include <stdio.h>
#include "sgx_eid.h"
#include "sgx_urts.h"
#include "sgx_error.h" 
#include "hybridsgx_util.h"

#include <iostream>
#include <string>

// #include "globals.h"
#include "util/connect.h"

#include "util/Profiler.h"
// #include "util/model.h"
// #include "util/util.cuh"
#include "ext/cxxopts.hpp"
#include <json.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "alexnet.h"
#include "alexnet.c"
#include "npy.hpp"
#include <omp.h>
#include <time.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include "data/cifar/cifar10_reader.hpp"
#define FILE_TRAIN_IMAGE		"/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/data/train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/data/train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/data/t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10_cifar10/server_sgx/ServerSGX/data/train-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		10000
#define COUNT_TEST		600
#define BATCHSIZE       1000


int partyNum=0;
int total_parties=2;
// std::vector<AESObject*> aes_objects;
//AESObject* aes_indep;
//AESObject* aes_next;
//AESObject* aes_prev;
//Precompute PrecomputeObject;

extern std::string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern Profiler matmul_profiler;
Profiler func_profiler;
Profiler memory_profiler;
Profiler comm_profiler;
Profiler debug_profiler;

nlohmann::json piranha_config;

size_t db_bytes = 0;
size_t db_layer_max_bytes = 0;
size_t db_max_bytes = 0;

size_t train_dataset_size = 60000;
size_t test_dataset_size = 10000;
int log_learning_rate = 5;
size_t INPUT_SIZE;
size_t NUM_CLASSES;

sgx_enclave_id_t global_server_eid;
int total_relu_time=0;
int total_grad_relu_time=0;
int total_maxpool_time=0;
int total_grad_maxpool_time=0;
decltype(std::chrono::system_clock::now()) start;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration;
decltype(std::chrono::system_clock::now()) start_1epoch;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_1epoch;
decltype(std::chrono::system_clock::now()) start_relu;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_relu;
decltype(std::chrono::system_clock::now()) start_gradrelu;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_gradrelu;
decltype(std::chrono::system_clock::now()) start_maxpool;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_maxpool;
decltype(std::chrono::system_clock::now()) start_gradmaxpool;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_gradmaxpool;
void start_timer_maxpool() {
	start_maxpool = std::chrono::system_clock::now();
}
void stop_timer_maxpool() {
	duration_maxpool = std::chrono::system_clock::now() - start_maxpool;
    std::cout << "The time of a maxpool is: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_maxpool).count() << std::endl;
    total_maxpool_time+=std::chrono::duration_cast<std::chrono::milliseconds>(duration_maxpool).count();
}

void start_timer_grad_maxpool() {
	start_gradmaxpool = std::chrono::system_clock::now();
}
void stop_timer_grad_maxpool() {
	duration_gradmaxpool = std::chrono::system_clock::now() - start_gradmaxpool;
    std::cout << "The time of a grad_maxpool is: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_gradmaxpool).count() << std::endl;
    total_grad_maxpool_time+=std::chrono::duration_cast<std::chrono::milliseconds>(duration_gradmaxpool).count();
}


void start_timer_relu() {
	start_relu = std::chrono::system_clock::now();
}

void stop_timer_relu() {
	duration_relu = std::chrono::system_clock::now() - start_relu;
      std::cout << "The time of a relu is: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_relu).count() << std::endl;
    total_relu_time+=std::chrono::duration_cast<std::chrono::milliseconds>(duration_relu).count();
}
void start_timer_grad_relu() {
	start_gradrelu = std::chrono::system_clock::now();
}

void stop_timer_grad_relu() {
	duration_gradrelu = std::chrono::system_clock::now() - start_gradrelu;
      std::cout << "The time of a grad_relu is: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_gradrelu).count() << std::endl;
    total_grad_relu_time+=std::chrono::duration_cast<std::chrono::milliseconds>(duration_gradrelu).count();
}


void start_timer() {
	start = std::chrono::system_clock::now();
}
void stop_timer() {
	duration = std::chrono::system_clock::now() - start;
}
void start_timer_1epoch() {
	start_1epoch = std::chrono::system_clock::now();
}
void stop_timer_1epoch() {
	duration_1epoch = std::chrono::system_clock::now() - start;
}


#define VECTOR_DATA_W_FILE(data, shapen, datav,shapev,file)    \
{                                                         \
    datav.clear();                                        \
    shapev.clear();                                       \
    datav.resize(sizeof(data));                           \
    shapen[0]=GETLENGTH(data);                             \ 
    shapen[1]=GETLENGTH(*data);                             \
    shapen[2]=GETLENGTH(**data);                            \
    shapen[3]=GETLENGTH(***data);                           \
    std::vector<unsigned long> shapev(shapen,shapen+4);     \
    memcpy(&datav[0],data,BATCHSIZE*sizeof(data[0]));       \
    const npy::npy_data<double> data_npy{datav, shapev, false};\
    write_npy(file, data_npy);                                  \
}                                                               \

#define VECTOR_WEIGHT_W_FILE(weight,shapewn, weightv,shapewv,file)  \
{                                                                     \
    weightv.clear();                                              \
    shapewv.clear();                                               \
    weightv.resize(GETCOUNT(weight));                                   \
    shapewn[0]=GETLENGTH(weight);                                   \ 
    shapewn[1]=GETLENGTH(*weight);                                   \
    shapewn[2]=GETLENGTH(**weight);                                   \
    shapewn[3]=GETLENGTH(***weight);                                 \
    std::vector<unsigned long> shapewv(shapewn,shapewn+4);          \
    memcpy(&weightv[0],weight, sizeof(weight));                     \
    const npy::npy_data<double> weight_npy{weightv, shapewv, false};\
    write_npy(file, weight_npy);                                    \
}                                                                   \        

#define VECTOR_BIAS_W_FILE(bias,shapebn, biasv,shapebv,file)        \
{                                                                   \
    biasv.clear();                                                  \
    shapebv.clear();                                                \
    biasv.resize(GETCOUNT(bias));                                     \
    shapebn[0]=GETLENGTH(bias);                                     \ 
    std::vector<unsigned long> shapebv(shapebn,shapebn+1);          \
    memcpy(&biasv[0],bias, sizeof(bias));                           \
    const npy::npy_data<double> bias_npy{biasv, shapebv, false};    \
    write_npy(file, bias_npy);                                      \
}             

#define VECTOR_FC_W_FILE(data,shapefn, datav,shapefv,file)        \
{                                                                   \
    datav.clear();                                                  \
    shapefv.clear();                                                \
    datav.resize(GETCOUNT(data));                                     \
    shapefn[0]=GETLENGTH(data);                                     \
    shapefn[1]=GETLENGTH(*data);                                    \
    std::vector<unsigned long> shapefv(shapefn,shapefn + 2);         \
    memcpy(&datav[0],data, sizeof(data));                           \
    const npy::npy_data<double> data_fc_npy{datav, shapefv, false};    \
    write_npy(file, data_fc_npy);                                      \
}                                                                   \   


#define VECTOR_FCData_W_FILE(data,shapefn, datav,shapefv,file)        \
{                                                                   \
    datav.clear();                                                  \
    shapefv.clear();                                                \
    datav.resize(GETCOUNT(data));                                     \
    shapefn[0]=BATCHSIZE;                                              \
    shapefn[1]=datav.size()/shapefn[0];                             \
    std::vector<unsigned long> shapefv(shapefn,shapefn + 2);         \
    memcpy(&datav[0],data, datav.size()*sizeof(double));                           \
    const npy::npy_data<double> data_fc_npy{datav, shapefv, false};    \
    write_npy(file, data_fc_npy);                                      \
}                                                                   \              

#define READ_PY(data,file)                                          \
{                                                                   \
    data_py.clear();                                                \
    shape_py.clear();                                               \
    npy::LoadArrayFromNumpy(file, shape_py, is_fortran, data_py);   \
    data =(double *)malloc(data_py.size()*sizeof(double));          \
    memcpy(data,&data_py[0],data_py.size()*sizeof(double));         \
}                                                                   \                



static inline void load_input_four(Feature *features, image input,int n)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input[n];
	const long sz = 32*32;
	double mean[3]={0}, std[3] = {0};
	FOREACH(i,3)
	{
		FOREACH(j, 32)
			FOREACH(k, 32)
		{
			mean[i] += input[i][j][k];
			std[i] += input[i][j][k] * input[i][j][k];
		}
		mean[i] /= sz;
		std[i] = sqrt(std[i] / sz - mean[i]*mean[i]);
	}

	FOREACH(i,3)
	FOREACH(j, 32)
		FOREACH(k, 32)
	{
		layer0[i][j][k] = (input[i][j][k] - mean[i]) / std[i];
	}
	// printf("input[0][0]:%u\n",input[0][0]);
	// printf("layer0[0][0]:%f\n",layer0[0][0]);
	//sleep(30);
}
static void load_target_four(Feature *features, Feature *errors, int label,int n)
{
	double *output = (double *)features->output[n];
	double *error = (double *)errors->output[n];
	softmax(output, error, label, GETCOUNT(features->output[n]));
}
void grad_relu(int global_server_eid, double *input, size_t n){
    double *data_out_sgx=(double *)malloc(n*sizeof(double));
    double *data_sgx=(double *)malloc(n*sizeof(double));
    memcpy(data_sgx, input, n*sizeof(double));
    start_timer_grad_relu();
    grad_relu_sgx(global_server_eid,data_out_sgx,data_sgx,n);
    stop_timer_grad_relu();
    memcpy(input, data_out_sgx,n*sizeof(double));
}
void grad_maxpool(int global_server_eid, double *outerror, double *inerror, double *input,int C, int input_length, size_t n_out, size_t n_in){
    double *data_outerror=(double *)malloc(n_out*sizeof(double));
    double *data_inerror=(double *)malloc(n_in*sizeof(double));
    double *data_input=(double *)malloc(n_in*sizeof(double));
    int shape_input[4]={BATCHSIZE,C,input_length,input_length};
    memcpy(data_input,input,n_in*sizeof(double));
    memcpy(data_outerror,outerror,n_out*sizeof(double));
    start_timer_grad_maxpool();
    grad_maxpool_2_sgx(global_server_eid,data_outerror,data_inerror,data_input,shape_input,n_out,n_in);
    stop_timer_grad_maxpool();
    memcpy(inerror,data_inerror,n_in*sizeof(double));
}

void grad_stride_maxpool(int global_server_eid, double *outerror, double *inerror, double *input,int C, int input_length, size_t n_out, size_t n_in,size_t kernel, size_t stride){
    double *data_outerror=(double *)malloc(n_out*sizeof(double));
    double *data_inerror=(double *)malloc(n_in*sizeof(double));
    double *data_input=(double *)malloc(n_in*sizeof(double));
    int shape_input[4]={BATCHSIZE,C,input_length,input_length};
    memcpy(data_input,input,n_in*sizeof(double));
    memcpy(data_outerror,outerror,n_out*sizeof(double));
    start_timer_grad_maxpool();
    grad_maxpool_str_sgx(global_server_eid,data_outerror,data_inerror,data_input,shape_input,n_out,n_in,kernel,stride);
    stop_timer_grad_maxpool();
    memcpy(inerror,data_inerror,n_in*sizeof(double));
}

int main(int argc){
   // Parse options -- retrieve party id and config JSON
//    printf("kkkkkkkkkkkkkkkkk\n");
    cxxopts::Options options("piranha", "GPU-accelerated platform for MPC computation");
    options.add_options()
        ("p,party", "Party number", cxxopts::value<int>());
    // printf("0000000000000000000000000000000000\n");
    // options.allow_unrecognised_options();

    // auto parsed_options = options.parse(argc, argv);
    // printf("111111111111111111111111111111");
    // // Print help
    // if (parsed_options.count("help")) {
    //     std::cout << options.help() << std::endl;
    //     std::cout << "Report bugs to jlw@berkeley.edu" << std::endl;
    //     return 0;
    // }
    // printf("2222222222222222222222222222");
    // partyNum = parsed_options["party"].as<int>();

    // std::ifstream input_config(parsed_options["config"].as<std::string>());
    // input_config >> piranha_config;

    // // Start memory profiler and initialize communication between parties
    // memory_profiler.start();

    // //XXX initializeCommunication(options.ip_file, partyNum);

/*SGX enclave 初始化*/
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_server_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    printf("SGX class initial enclave of eid: %ld here\n", global_server_eid);

    std::vector<std::string> party_ips;
    for (int i = 0; i < total_parties; i++) {
	    party_ips.push_back("127.0.0.1");
    }
    // /*通信初始化*/
    // initializeCommunication(party_ips, partyNum, total_parties);
    // std::cout<<"Partynum:"<<partyNum<<std::endl;
    // printf("Initializing connection........\n");
    // start_timer();
    // std::vector<double> recev(1000);
    // receiveVector(1,recev);
    // stop_timer();
    // std::cout << "time of data receiving in non-secure world: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;
    // //synchronize(10000, 2); // wait for everyone to show up :)
    // sleep(3);

    // system("python test_GPU.py");

    std::string weight_npy_file="/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/weight.npy";
    std::string bias_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/bias.npy";
    std::string data_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/data.npy";
    std::string outerror_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/outerror.npy";
    std::string wd_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/weight_delta.npy";
    std::string bd_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/bias_delta.npy";
    std::string input_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/input.npy";

    std::string inerror_npy_file = "/home/jin/2023_11_22_CPU_GPU/AlexNet_GPU_SGX_without_com_cifar10/server_sgx/ServerSGX/global_memory/inerror.npy";



    std::vector<double> partyA(100000,1);
    std::vector<float> partyB(100,6);
    partyA[0]=2;
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));


    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    /*读取数据*/
    memcpy(train_data,&dataset.training_images[0],COUNT_TRAIN*sizeof(image));
    memcpy(train_label,&dataset.training_labels[0],COUNT_TRAIN*sizeof(uint8));
    std::cout<<"The size of dataset.training_images is:"<<dataset.training_images.size()<<std::endl;//读取所有的5万条数据
    std::cout<<"The size of dataset.training_images[0] is:"<<dataset.training_images[0].size()<<std::endl;//每条数据大小
    AlexNet *alexnet = (AlexNet *)malloc(sizeof(AlexNet));
	Initial(alexnet);
    Feature features={0};
    int iteration=0;
for(int ba=0;ba<=COUNT_TRAIN-BATCHSIZE;ba+=BATCHSIZE){
    iteration+=1;
    printf("Here is the %f batch\n",(float)ba/COUNT_TRAIN);
#pragma omp parallel for
        for(int i=0;i<BATCHSIZE;i++){
                load_input_four(&features, (train_data+0)[i],i);
        }
// #pragma omp parallel for
//         for(int i=0;i<BATCHSIZE;i++){
//             for(int j=0;j<GETCOUNT(features.input[0][0]);j++){
//                 printf("%d in batch 0, j:%d,value:%f, thread:%d thread total:%d\n",i,j,((double *)features.input[i][0])[j],omp_get_thread_num(),omp_get_num_threads());
//             }
//         }

/* C++ NPY转换*/
    /*AlexNet第一层输入数据读取到文件*/
    std::vector<double> data_v;
    unsigned long shape_n[4];
    std::vector<unsigned long> shape_v;
    VECTOR_DATA_W_FILE(features.input,shape_n,data_v,shape_v,data_npy_file);
    /*AlexNet第一层权重读取到文件*/
    std::vector<double> weight_v;
    unsigned long shape_wn[4];
    std::vector<unsigned long> shape_wv;
    VECTOR_WEIGHT_W_FILE(alexnet->weight1,shape_wn,weight_v,shape_wv,weight_npy_file);
    /*AlexNet第一层bias读取到文件*/
    std::vector<double> bias_v;
    unsigned long shape_bn[1];
    std::vector<unsigned long> shape_bv;
    VECTOR_BIAS_W_FILE(alexnet->bias1,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*送到GPU计算第一层卷积*/
    system("python conv_GPU.py");
    
    /*读取GPU第一层卷积计算后的结果*/
    std::vector<unsigned long> shape_py;
    std::vector<double> data_py; // 必须指定<dtype>类型与npy对应
    bool is_fortran;
    double *data_sgx=(double *)malloc(GETCOUNT(features.layer1_conv)*sizeof(double));
    READ_PY(data_sgx,data_npy_file);
    double *data_out_sgx=(double *)malloc(data_py.size()*sizeof(double));
    /*Intel SGX 第一层relu calculation*/
    start_timer_relu();
    relu_sgx(global_server_eid, data_out_sgx,data_sgx,data_py.size());
    stop_timer_relu();
	std::cout << "time of relu computation in  sgx secure world: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;

    /*记录第一层计算结果到数组*/
    memcpy(features.layer1_conv,data_out_sgx,data_py.size()*sizeof(data_py[0]));
    
    /*计算第一层maxpool在SGX里*/
    int shape_sgx[4]={BATCHSIZE,LAYER1,LENGTH_FEATURE1_1,LENGTH_FEATURE1_1};
    size_t maxpool_out_size = data_py.size()/4;
    data_sgx=(double *)malloc(data_py.size()*sizeof(double));
    memcpy(data_sgx,features.layer1_conv,data_py.size()*sizeof(double));
    data_out_sgx=(double *)malloc(maxpool_out_size*sizeof(double));
    start_timer_maxpool();
    maxpool_2_sgx(global_server_eid,data_out_sgx,data_sgx,shape_sgx,maxpool_out_size, data_py.size());
    stop_timer_maxpool();
    /*第一层maxpool计算结果记录到features数组*/
    memcpy(features.layer1_pool,data_out_sgx, maxpool_out_size*sizeof(double));

    /*第一层maxpool计算结果读取到文件*/
    VECTOR_DATA_W_FILE(features.layer1_pool,shape_n,data_v,shape_v,data_npy_file);
    /*第二层权重读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight2,shape_wn,weight_v,shape_wv,weight_npy_file);
    /*第二层bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias2,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*第二层GPU计算*/
    system("python conv_GPU.py");
    /*读取第二层卷积计算结果,并在SGX里计算激活函数*/
    READ_PY(data_sgx,data_npy_file);
    data_out_sgx =(double *)malloc(data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_out_sgx,data_sgx,data_py.size());
    stop_timer_relu();
    /*第二层计算结果存储到features数组*/
    memcpy(features.layer2_conv,data_out_sgx,data_py.size()*sizeof(double));

    /*SGX里第二层池化计算*/
    data_sgx=(double *)malloc(data_py.size()*sizeof(double));
    maxpool_out_size=data_py.size()/4;
    data_out_sgx=(double *)malloc(maxpool_out_size*sizeof(double));
    memcpy(data_sgx,features.layer2_conv,data_py.size()*sizeof(double));
    shape_sgx[0]=BATCHSIZE;shape_sgx[1]=LAYER2;shape_sgx[2]=LENGTH_FEATURE2_1;shape_sgx[3]=LENGTH_FEATURE2_1;
    start_timer_maxpool();
    maxpool_2_sgx(global_server_eid,data_out_sgx,data_sgx,shape_sgx,maxpool_out_size,data_py.size());
    stop_timer_maxpool();
    /*第二层池化计算存储到数组*/
    memcpy(features.layer2_pool,data_out_sgx,maxpool_out_size*sizeof(double));
    /*第二层maxpool计算结果读取到文件*/
    VECTOR_DATA_W_FILE(features.layer2_pool,shape_n,data_v,shape_v,data_npy_file);
    /*第三层权重读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight3,shape_wn,weight_v,shape_wv,weight_npy_file);
    /*第三层bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias3,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*第三层卷积计算*/
    system("python conv_GPU.py");

    /*读取第三层卷积计算，并在SGX里计算激活函数*/
    READ_PY(data_sgx,data_npy_file);
    start_timer_relu();
    relu_sgx(global_server_eid,data_sgx,data_sgx,data_py.size());
    stop_timer_relu();
    /*第三层计算结果存储到数组*/
    memcpy(features.layer3_conv,data_sgx,data_py.size()*sizeof(double));

    /*第四层卷积计算*/
    /*第四层卷积input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer3_conv,shape_n,data_v,shape_v,data_npy_file);
    /*第四层卷积weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight4,shape_wn,weight_v,shape_wv,weight_npy_file);
    /*第四层卷积weight读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias4,shape_bn,bias_v,shape_bv,bias_npy_file);
    system("python conv_GPU.py");
    /*读取第四层卷积结果到数组，并做激活运算*/
    READ_PY(data_sgx,data_npy_file);
    memcpy(features.layer4_conv,data_sgx,data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_sgx,data_sgx,data_py.size());
    stop_timer_relu();
    /*第五层卷积*/
    /*第五层卷积input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer4_conv,shape_n,data_v,shape_v,data_npy_file);
    /*第五层卷积weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight5,shape_wn,weight_v,shape_wv,weight_npy_file);
    /*第五层卷积weight读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias5,shape_bn,bias_v,shape_bv,bias_npy_file);
    system("python conv_GPU.py");
    /*读取第五层卷积结果到数组，并做激活运算*/
    READ_PY(data_sgx,data_npy_file);
    data_out_sgx =(double *)malloc(data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_out_sgx,data_sgx,data_py.size());
    stop_timer_relu();
    memcpy(features.layer5_conv,data_out_sgx,data_py.size()*sizeof(double));

    /*第五层池化*/
    size_t n_in=data_py.size();
    int kernel=3;
    int stride=2;
    size_t n_out=BATCHSIZE*LAYER5*LENGTH_FEATURE5_2*LENGTH_FEATURE5_2;
    // data_out_sgx=(double *)malloc(n_out*sizeof(double));
    // data_sgx=(double *)malloc(n_in*sizeof(double));
    shape_sgx[0]=BATCHSIZE;shape_sgx[1]=LAYER5;shape_sgx[2]=LENGTH_FEATURE5_1;shape_sgx[3]=LENGTH_FEATURE5_1;
    // maxpool_out_size=data_py.size()/4;
    // printf("The data pt size is%d\n",data_py.size)//16384
    memcpy(data_sgx, features.layer5_conv,n_in*sizeof(double));
    start_timer_maxpool();
    maxpool_2_sgx(global_server_eid,data_out_sgx,data_sgx,shape_sgx,data_py.size()/4, data_py.size());
    stop_timer_maxpool();

    /*第六层 FC1 计算*/
    /*第六层 FC1 输入写到文件*/
    std::vector<unsigned long> shape_fv(2);
    int shape_fn[2];
    VECTOR_FCData_W_FILE(features.layer5_pool,shape_fn,data_v,shape_fv,data_npy_file);
    // VECTOR_DATA_W_FILE(features.layer5_pool,shape_n,data_v,shape_v,data_npy_file);
    /*第六层 FC1 计算 第六层权重读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc1,shape_fn,data_v,shape_fv,weight_npy_file);
    /*第六层 FC1 计算 第六层bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc1,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python fc_GPU.py");
    /*第六层 FC1 计算结果读取 并在SGX里计算激活函数*/
    READ_PY(data_sgx, data_npy_file);
    data_out_sgx =(double *)malloc(data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_out_sgx,data_sgx,data_py.size());;
    stop_timer_relu();
    memcpy(features.fc1,data_out_sgx,data_py.size()*sizeof(double));
    /*第六层 FC2 计算*/
    /*第六层 FC2 输入写到文件*/
    VECTOR_FC_W_FILE(features.fc1,shape_fn,data_v,shape_fv,data_npy_file);
    /*第六层 FC2 计算 第六层权重读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc2,shape_fn,data_v,shape_fv,weight_npy_file);
    /*第六层 FC2 计算 第六层bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc2,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python fc_GPU.py");
    /*第六层 FC2 计算结果读取 并在SGX里计算激活函数*/
    READ_PY(data_sgx, data_npy_file);
    data_out_sgx =(double *)malloc(data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_out_sgx,data_sgx,data_py.size());;
    stop_timer_relu();
    memcpy(features.fc2,data_sgx,data_py.size()*sizeof(double));

    /*第六层 FC3 计算*/
    /*第六层 FC3 输入写到文件*/
    VECTOR_FC_W_FILE(features.fc2,shape_fn,data_v,shape_fv,data_npy_file);
    /*第六层 FC3 计算 第六层权重读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc3,shape_fn,data_v,shape_fv,weight_npy_file);
    /*第六层 FC3 计算 第六层bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc3,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python fc_GPU.py");
    /*第六层 FC3 计算结果读取*/
    READ_PY(data_sgx, data_npy_file);
    data_out_sgx =(double *)malloc(data_py.size()*sizeof(double));
    start_timer_relu();
    relu_sgx(global_server_eid,data_out_sgx,data_sgx,data_py.size());;
    stop_timer_relu();
    memcpy(features.output,data_sgx,data_py.size()*sizeof(double));




    /*后向传导*/
    Feature errors = { 0 };
    /*Loss 计算*/
    #pragma omp parallel for
        for(int i=0;i<BATCHSIZE;i++){
                load_target_four(&features,&errors,(train_label+0)[i],i);
        }

    /*第六层FC3 后向传导*/
    grad_relu(global_server_eid,((double*)errors.output),GETCOUNT(errors.output));
    /*outerror读取到文件*/
    VECTOR_FC_W_FILE(errors.output,shape_fn,data_v,shape_fv,outerror_npy_file);
    /*input 读取到文件*/
    VECTOR_FC_W_FILE(features.fc2,shape_fn,data_v,shape_fv,input_npy_file);
    /*weight读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc3,shape_fn,data_v,shape_fv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc3,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python grad_fc.py");

    /*读取GPU计算的FC2 error npy*/
    READ_PY(data_sgx, inerror_npy_file);
    memcpy(errors.fc2, data_sgx, data_py.size()*sizeof(double));
    /*读取GPU计算的output层权重weight update */
    READ_PY(data_sgx, weight_npy_file);
    memcpy(alexnet->fc3,data_sgx,data_py.size()*sizeof(double));
    /*读取GPU计算的output层偏置bias*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias_fc3,data_sgx,data_py.size()*sizeof(double));


    /*FC2 outerror读取到文件*/
    grad_relu(global_server_eid,((double*)errors.fc2),GETCOUNT(errors.fc2));
    VECTOR_FC_W_FILE(errors.fc2,shape_fn,data_v,shape_fv,outerror_npy_file);
    /*input 读取到文件*/
    VECTOR_FC_W_FILE(features.fc1,shape_fn,data_v,shape_fv,input_npy_file);
    /*FC2 weight读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc2,shape_fn,data_v,shape_fv,weight_npy_file);
    /*FC2 bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc2,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python grad_fc.py");

    /*读取GPU计算的FC1 error npy*/
    READ_PY(data_sgx, inerror_npy_file);
    memcpy(errors.fc1, data_sgx, data_py.size()*sizeof(double));
    /*读取GPU计算的output层权重weight update */
    READ_PY(data_sgx, weight_npy_file);
    memcpy(alexnet->fc2,data_sgx,data_py.size()*sizeof(double));
    /*读取GPU计算的output层偏置bias*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias_fc2,data_sgx,data_py.size()*sizeof(double));


    /*FC1 outerror读取到文件*/
    grad_relu(global_server_eid,((double*)errors.fc1),GETCOUNT(errors.output));
    VECTOR_FC_W_FILE(errors.fc1,shape_fn,data_v,shape_fv,outerror_npy_file);
    /*input 读取到文件*/
    VECTOR_FCData_W_FILE(features.layer5_pool,shape_fn,data_v,shape_fv,input_npy_file);
    /*FC1 weight读取到文件*/
    VECTOR_FC_W_FILE(alexnet->fc1,shape_fn,data_v,shape_fv,weight_npy_file);
    /*FC1 bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias_fc1,shape_fn,data_v,shape_fv,bias_npy_file);
    system("python grad_fc.py");
    /*读取GPU计算的FC1 error npy*/
    READ_PY(data_sgx, inerror_npy_file);
    memcpy(errors.layer5_pool, data_sgx, data_py.size()*sizeof(double));
    /*读取GPU计算的output层权重weight update */
    READ_PY(data_sgx, weight_npy_file);
    memcpy(alexnet->fc1,data_sgx,data_py.size()*sizeof(double));
    /*读取GPU计算的output层偏置bias*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias_fc1,data_sgx,data_py.size()*sizeof(double));


    /*第五层重合maxpool计算,layer5_conv error计算 并记录到数组*/
    n_in=GETCOUNT(features.layer5_conv);
    n_out=GETCOUNT(features.layer5_pool);
    grad_stride_maxpool(global_server_eid,((double*)errors.layer5_pool),((double*)errors.layer5_conv),((double*)features.layer5_conv),LAYER5,LENGTH_FEATURE5_1,n_out,n_in,kernel=3,stride=2);

    /*计算第五层卷积inerror of layer4_conv  error weight5*/
    /*SGX里计算第五层卷积grad relu conv outerrors*/
    grad_relu(global_server_eid,((double*)errors.layer5_conv),GETCOUNT(errors.layer5_conv));
    /*outerror读取到文件*/
    VECTOR_DATA_W_FILE(errors.layer5_conv,shape_n, data_v,shape_v,outerror_npy_file);
    /*weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight5,shape_wn, weight_v,shape_wv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias5,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer4_conv,shape_n, data_v,shape_v,input_npy_file);
    system("python grad_conv.py");
    /*layer4_conv inerror记录到数组*/
    READ_PY(data_sgx,inerror_npy_file);
    memcpy(errors.layer4_conv, data_sgx, data_py.size()*sizeof(double));
    /*第三层weight update记录到数组*/
    READ_PY(data_sgx,weight_npy_file);
    memcpy(alexnet->weight5, data_sgx, data_py.size()*sizeof(double));
    /*第三层bias update记录到数组*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias5, data_sgx, data_py.size()*sizeof(double));

    // /*第三层卷积error layer3_conv 计算weight4*/
    /*SGX里计算第四层卷积grad relu conv outerrors*/
    grad_relu(global_server_eid,((double*)errors.layer4_conv),GETCOUNT(errors.layer4_conv));
    /*outerror读取到文件*/
    VECTOR_DATA_W_FILE(errors.layer4_conv,shape_n, data_v,shape_v,outerror_npy_file);
    /*weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight4,shape_wn, weight_v,shape_wv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias4,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer3_conv,shape_n, data_v,shape_v,input_npy_file);
    system("python grad_conv.py");
    /*layer3_conv inerror记录到数组*/
    READ_PY(data_sgx,inerror_npy_file);
    memcpy(errors.layer3_conv, data_sgx, data_py.size()*sizeof(double));
    /*第三层weight update记录到数组*/
    READ_PY(data_sgx,weight_npy_file);
    memcpy(alexnet->weight4, data_sgx, data_py.size()*sizeof(double));
    /*第三层bias update记录到数组*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias4, data_sgx, data_py.size()*sizeof(double));




    /*第三层卷积inerror layer2_pool 计算 weight3*/
    /*SGX里计算第四层卷积grad relu conv outerrors*/
    grad_relu(global_server_eid,((double*)errors.layer3_conv),GETCOUNT(errors.layer4_conv));
    /*outerror读取到文件*/
    VECTOR_DATA_W_FILE(errors.layer3_conv,shape_n, data_v,shape_v,outerror_npy_file);
    /*weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight3,shape_wn, weight_v,shape_wv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias3,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer2_pool,shape_n, data_v,shape_v,input_npy_file);
    system("python grad_conv.py");
    /*layer2_pool inerror记录到数组*/
    READ_PY(data_sgx,inerror_npy_file);
    memcpy(errors.layer2_pool, data_sgx, data_py.size()*sizeof(double));
    /*第三层weight update记录到数组*/
    READ_PY(data_sgx,weight_npy_file);
    memcpy(alexnet->weight3, data_sgx, data_py.size()*sizeof(double));
    /*第三层bias update记录到数组*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias3, data_sgx, data_py.size()*sizeof(double));

    /*第二层maxpool layer2_conv error后向计算 并记录到数组*/
   grad_maxpool(global_server_eid,((double*)errors.layer2_pool),((double*)errors.layer2_conv),((double*)features.layer2_conv),GETLENGTH(*errors.layer2_conv),GETLENGTH(**errors.layer2_conv),GETCOUNT(errors.layer2_pool),GETCOUNT(errors.layer2_conv));
    /*第二层 conv layer1_pool layer计算*/
    /*SGX里计算第二层卷积grad layer2_conv relu conv outerrors*/
    grad_relu(global_server_eid,((double*)errors.layer2_conv),GETCOUNT(errors.layer2_conv));
    /*outerror读取到文件*/
    VECTOR_DATA_W_FILE(errors.layer2_conv,shape_n, data_v,shape_v,outerror_npy_file);
    /*weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight2,shape_wn, weight_v,shape_wv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias2,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*input读取到文件*/
    VECTOR_DATA_W_FILE(features.layer1_pool,shape_n, data_v,shape_v,input_npy_file);
    system("python grad_conv.py");
    /*layer2_pool inerror记录到数组*/
    READ_PY(data_sgx,inerror_npy_file);
    memcpy(errors.layer1_pool, data_sgx, data_py.size()*sizeof(double));
    /*第三层weight update记录到数组*/
    READ_PY(data_sgx,weight_npy_file);
    memcpy(alexnet->weight2, data_sgx, data_py.size()*sizeof(double));
    /*第三层bias update记录到数组*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias2, data_sgx, data_py.size()*sizeof(double));


    
    /*第一层maxpool layer1_conv error后向计算 并记录到数组*/
   grad_maxpool(global_server_eid,((double*)errors.layer1_pool),((double*)errors.layer1_conv),((double*)features.layer1_conv),GETLENGTH(*errors.layer1_conv),GETLENGTH(**errors.layer1_conv),GETCOUNT(errors.layer1_pool),GETCOUNT(errors.layer1_conv));
    /*第一层 conv input error layer计算*/
    /*SGX里计算第二层卷积grad layer2_conv relu conv outerrors*/
    grad_relu(global_server_eid,((double*)errors.layer1_conv),GETCOUNT(errors.layer1_conv));
    /*outerror读取到文件*/
    VECTOR_DATA_W_FILE(errors.layer1_conv,shape_n, data_v,shape_v,outerror_npy_file);
    /*weight读取到文件*/
    VECTOR_WEIGHT_W_FILE(alexnet->weight1,shape_wn, weight_v,shape_wv,weight_npy_file);
    /*bias读取到文件*/
    VECTOR_BIAS_W_FILE(alexnet->bias1,shape_bn,bias_v,shape_bv,bias_npy_file);
    /*input读取到文件*/
    VECTOR_DATA_W_FILE(features.input,shape_n, data_v,shape_v,input_npy_file);
    system("python grad_conv.py");
    /*layer2_pool inerror记录到数组*/
    READ_PY(data_sgx,inerror_npy_file);
    memcpy(errors.input, data_sgx, data_py.size()*sizeof(double));
    /*第三层weight update记录到数组*/
    READ_PY(data_sgx,weight_npy_file);
    memcpy(alexnet->weight1, data_sgx, data_py.size()*sizeof(double));
    /*第三层bias update记录到数组*/
    READ_PY(data_sgx, bias_npy_file);
    memcpy(alexnet->bias1, data_sgx, data_py.size()*sizeof(double));
    
    std::cout<<"relu time average each iteration:"<<(float)total_relu_time/iteration<<std::endl;
    std::cout<<"grad relu time average each iteration:"<<(float)total_grad_relu_time/iteration<<std::endl;
    std::cout<<"maxpool time average each iteration:"<<(float)total_maxpool_time/iteration<<std::endl;
    std::cout<<"grad maxpool time average each iteration:"<<(float)total_grad_maxpool_time/iteration<<std::endl;

}
    /*一个batch的前向传导和梯度更新结束*/
    sgx_destroy_enclave(global_server_eid);
    return 0;
 }

