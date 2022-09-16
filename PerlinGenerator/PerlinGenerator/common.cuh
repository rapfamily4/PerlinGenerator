#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#ifndef _COMMON_H
#define _COMMON_H


/* Macro useful for CUDA error checking. */
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

/* It prints the linear array of the given size; for debugging purposes. */
void printArray(int *array, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        if (i != 0) printf(", ");
        printf("%d", array[i]);
    }
    printf("]");
}

/* It prints the device name. */
inline void printdeviceName() {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}


#endif