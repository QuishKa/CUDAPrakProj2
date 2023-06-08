
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define MATSIZE 5
#define BLOCK_SIZE 32

const float gaussRef[MATSIZE * MATSIZE] = {
        0.000789, 0.006581, 0.013347, 0.006581, 0.000789,
        0.006581, 0.054901, 0.111345, 0.054901, 0.006581,
        0.013347, 0.111345, 0.225821, 0.111345, 0.013347,
        0.006581, 0.054901, 0.111345, 0.054901, 0.006581,
        0.000789, 0.006581, 0.013347, 0.006581, 0.000789
};

//const float gaussRef[5][5] = {
//        {0, 0, 0, 0, 0},
//        {0, 0.5, 0.75, 0.5, 0},
//        {0, 0.75, 1, 0.75, 0},
//        {0, 0.5, 0.75, 0.5, 0},
//        {0, 0, 0, 0, 0}
//};

cudaError_t gaussCUDA(unsigned char **img, const float *gauss, int w, int h);

__global__ void gaussKernel(unsigned char *ref, unsigned char *res, float *gauss, const int w, const int h)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char tmp = 0;
    if (row < h - 3 && col < w - 3) {
        for (int j = -2; j < 3; ++j) {
            for (int i = -2; i < 3; ++i) {
                tmp += (unsigned char)(gauss[(j + 2) * MATSIZE + i + 2] * ref[(row + j) * w + col + i]);
            }
        }
        res[row * w + col] = tmp;
    }
}

int main()
{
    // CPU implementation
    cv::Mat gauss = cv::Mat(MATSIZE, MATSIZE, CV_32FC1);
    for (int y = 0; y < MATSIZE; ++y) {
        for (int x = 0; x < MATSIZE; ++x) {
            gauss.at<float_t>(y, x) = gaussRef[MATSIZE * y + x];
        }
    }

    cv::Mat image;
    image = cv::imread("C:\\Users\\slava\\Desktop\\123.png", cv::IMREAD_GRAYSCALE);
    
    int w, h;
    w = image.cols;
    h = image.rows;
    for (int y = ceil(MATSIZE / 2); y < h - ceil(MATSIZE / 2); ++y) {
        for (int x = ceil(MATSIZE / 2); x < w - ceil(MATSIZE / 2); ++x) {
            float tmp = 0;
            for (int j = floor(MATSIZE / 2) * -1; j < ceil(MATSIZE / 2); ++j) {
                for (int i = floor(MATSIZE / 2) * -1; i < ceil(MATSIZE / 2); ++i) {
                    tmp += gauss.at<float_t>(j + floor(MATSIZE / 2), i + floor(MATSIZE / 2)) * image.at<uchar>(y + j, x + i);
                }
            }
            image.at<uchar>(y, x) = tmp;
        }
    }

    cv::imshow("Display Image", image);
    cv::waitKey(0);
    //

    // GPU implementation
    image = cv::imread("C:\\Users\\slava\\Desktop\\123.png", cv::IMREAD_GRAYSCALE);
    w = image.cols;
    h = image.rows;
    // use gaussRef
    // cv::Mat to uchar
    //unsigned char* img = (unsigned char*)image.data; // may not work XD
    cv::Mat flat = image.reshape(1, image.total() * image.channels());
    printf("image channels: %i\n", image.channels());
    if (!image.isContinuous()) {
        flat = image.clone();
    }
    unsigned char* img = flat.data;

    printf("pix 1,1 mat: %i\n", image.at<uchar>(0, 0));
    printf("pix 1,1 nemat: %i\n", img[0]);
    printf("pix 1,2 mat: %i\n", image.at<uchar>(0, 1));
    printf("pix 1,2 nemat: %i\n", img[1]);
    
    cudaError_t cudaStatus = gaussCUDA(&img, gaussRef, w, h);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    // uchar to cv::Mat
    cv::Mat res = cv::Mat(h, w, image.type(), img); // should work
    printf("pix 1,2 mat: %i\n", image.at<uchar>(0, 1));
    printf("pix 1,2 nemat: %i\n", img[1]);
    printf("pix 1,2 res: %i\n", res.at<uchar>(0, 1));

    cv::imshow("Display Image", res);
    cv::waitKey(0);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t gaussCUDA(unsigned char** img, const float* hgauss, int w, int h)
{
    const int size = w * h;
    unsigned char *htmp = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *dtmp, *dref, dw, dh;
    float* dgauss;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // allocate memory on device
    cudaStatus = cudaMalloc(&dtmp, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc tmp failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dref, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc tmp failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dgauss, MATSIZE * MATSIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc gauss failed!");
        goto Error;
    }

    // copy to device
    cudaStatus = cudaMemcpy(dtmp, *img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy tmp failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dref, *img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy tmp failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dgauss, hgauss, MATSIZE * MATSIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy gauss failed!");
        goto Error;
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(w / dimBlock.x, h / dimBlock.y);
    // Launch a kernel on the GPU with one thread for each element.
    gaussKernel<<<dimGrid, dimBlock>>>(dref, dtmp, dgauss, w, h);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(*img, dtmp, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result failed!");
        goto Error;
    }

Error:
    cudaFree(dtmp);
    cudaFree(dgauss);
    cudaFree(dref);
    
    return cudaStatus;
}
