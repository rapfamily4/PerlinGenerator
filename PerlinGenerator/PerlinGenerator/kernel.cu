#include "common.cuh"
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "stb_image_write.h"

#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Debug configuration */
#define DEBUG           0 // Enables debug configuration.
#define DEBUG_COMPARE   0 // Enables comparison of device and host codes; available only if DEBUG is enabled.

/* Runtime configuration */
#define BLOCK_SIZE_1D             256
#define BLOCK_SIZE_2D             16
#define STREAM_NUMBER             4
#define PINNED_MEMORY             1 // Enables pinned memory usage for storing the noise map on host.
#define WARP_UNROLLING            1 // Enables warp unrolling during the [min, max] range computation.
#if WARP_UNROLLING
#define STRIDE_LIMIT              32
#else
#define STRIDE_LIMIT              0
#endif

/* Generation constants */
#define DEFAULT_MAP_SIZE          1024
#define DEFAULT_NOISE_DENSITY     1
#define DEFAULT_DESIRED_OCTAVES   1
#define DEFAULT_FALLOFF           0.5f
#define DEFAULT_ABSOLUTE_VALUE    false
char    DEFAULT_OUTPUT_PATH[10] = "./map.png";


/* Include host version of Perlin Noise if host-device comparison is requested. */
#if DEBUG
#if DEBUG_COMPARE
#include "host_perlin.cuh"
#endif // DEBUG_COMPARE
#endif // DEBUG


/* Atomic add for floats. */
__device__ __forceinline__ float atomicAddFloat(float *address, float val) {
    // Source: https://github.com/tum-vision/tandem/issues/11
    int *address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val + __int_as_float(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __int_as_float(old);
}

/* Atomic min for floats. */
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    // Source: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

/* Atomic max for floats. */
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    // Source: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

/* Device function which linearly interpolates between two values. */
__device__ __forceinline__ float lerp(float v0, float v1, float t) {
    // Source: https://developer.nvidia.com/blog/lerp-faster-cuda/
    return fma(t, v1, fma(-t, v0, v0));
}

/* Kernel which generates the gradients. */
__global__ void generateGradients(float *gradients, int gradientsNumber, int totalGradients, int seed) {
    // Make sure the linear index is referring to a gradient.
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= gradientsNumber) return;

    // Initialize random generator.
    // NOTE: Each thread has its own random generator.
    // NOTE: The number of gradients to generate (gradientsNumber) is also used as an offset
    //       for the random generator; this should help to randomize the results even more.
    curandState_t curandState;
    curand_init(
        seed,            // Seed for the random generator.
        linearIdx,       // Sequence number: used to differ returned number among cores sharing the same seed.
        gradientsNumber, // Offset; can be zero.
        &curandState     // State of this core's random generator.   
    );

    // Calculate random gradient.
    float randomX;
    float randomY;
    randomX = -1.0f + curand_uniform(&curandState) * 2.0f;
    randomY = -1.0f + curand_uniform(&curandState) * 2.0f;

    // Store the coordinates in the gradients array.
    gradients[linearIdx] = randomX;
    gradients[linearIdx + totalGradients] = randomY;
}

/* Kernel handling the noise generation. */
__global__ void perlinNoise(float *map, float *gradients, int mapSize, int slabSize, int slabOffset, int noiseDensity, int octavesNumber, int gradientsNumber, float falloff, bool abs) {
    // Check sanity (is this pixel inside the map?).
    int mapX = blockIdx.x * blockDim.x + threadIdx.x;
    int mapY = blockIdx.y * blockDim.y + threadIdx.y + slabOffset;
    if (mapX >= mapSize || (mapY - slabOffset) >= slabSize) return;

    // Iterate among octaves.
    float finalHeight = 0.0f;
    float octaveWeight = 1.0f;
    int densityMultiplier = 1;
    int gradientsOffset = 0;
    for (int octave = 0; octave < octavesNumber; octave++) {
        // Calculate which cell of the gradient grid this pixel is in.
        int gradientsDensity = noiseDensity * densityMultiplier;
        int gradientsDistance = mapSize / gradientsDensity; // The distance between two gradients, in pixels.
        int gridX = mapX / gradientsDistance;
        int gridY = mapY / gradientsDistance;

        // Retrieve the four gradients nearest to this pixel.
        // NOTE: The way they are retrieved makes so that gradients shared among several pixels
        //       always return the same vector coordinates. This is crucial to achieve the
        //       relationship of a pixel with its neighborhood in terms of height values.
        float gradient[2][2][2];
        int octaveGradients = (gradientsDensity + 1) * (gradientsDensity + 1);
        // Top left.
        int tlIdx = (gridY * (gradientsDensity + 1) + gridX) % octaveGradients;           // linear index
        gradient[0][0][0] = gradients[tlIdx + gradientsOffset];                           // x
        gradient[0][0][1] = gradients[tlIdx + gradientsNumber + gradientsOffset];         // y
        // Top right.
        int trIdx = (gridY * (gradientsDensity + 1) + gridX + 1) % octaveGradients;       // linear index
        gradient[1][0][0] = gradients[trIdx + gradientsOffset];                           // x
        gradient[1][0][1] = gradients[trIdx + gradientsNumber + gradientsOffset];         // y
        // Bottom left.
        int blIdx = ((gridY + 1) * (gradientsDensity + 1) + gridX) % octaveGradients;     // linear index
        gradient[0][1][0] = gradients[blIdx + gradientsOffset];                           // x
        gradient[0][1][1] = gradients[blIdx + gradientsNumber + gradientsOffset];         // y
        // Bottom right.
        int brIdx = ((gridY + 1) * (gradientsDensity + 1) + gridX + 1) % octaveGradients; // linear index
        gradient[1][1][0] = gradients[brIdx + gradientsOffset];                           // x
        gradient[1][1][1] = gradients[brIdx + gradientsNumber + gradientsOffset];         // y

        // Calculate distance between the beginning of the grid's cell (top left) and the pixel.
        int deltaX = mapX - gradientsDistance * gridX;
        int deltaY = mapY - gradientsDistance * gridY;

        // Calculate height values of the gradients.
        // NOTE: These values are a dot product of:
        //           - the gradient;
        //           - the displacement vector (the vector from the gradient's origin to
        //             the pixel position).
        // NOTE: Such dot product will indicate how much the gradient affects the final
        //       height value of the pixel.
        float tlDot = (float)deltaX * gradient[0][0][0] + (float)deltaY * gradient[0][0][1];
        float trDot = (float)(deltaX - gradientsDistance) * gradient[1][0][0] + (float)deltaY * gradient[1][0][1];
        float blDot = (float)deltaX * gradient[0][1][0] + (float)(deltaY - gradientsDistance) * gradient[0][1][1];
        float brDot = (float)(deltaX - gradientsDistance) * gradient[1][1][0] + (float)(deltaY - gradientsDistance) * gradient[1][1][1];

        // Calculate the horizontal and vertical weights used for interpolation.
        // NOTE: The weight gets initialized as a linear weight, comparing the delta to the gradient distance.
        //       Then, apply a Smootherstep fuction on the weights for a smoother interpolation.
        float horWeight = (float)deltaX / (float)gradientsDistance;
        horWeight = (horWeight * (horWeight * 6.0 - 15.0) + 10.0) * horWeight * horWeight * horWeight;
        float verWeight = (float)deltaY / (float)gradientsDistance;
        verWeight = (verWeight * (verWeight * 6.0 - 15.0) + 10.0) * verWeight * verWeight * verWeight;

        // Interpolate values in order to get the height for this pixel.
        // NOTE: Interpolate twice horizontally, once vertically.
        float top = lerp(tlDot, trDot, horWeight);
        float bottom = lerp(blDot, brDot, horWeight);
        float octaveHeight = lerp(top, bottom, verWeight);

        // If desired, apply abs on height value.
        if (abs && octaveHeight < 0.0f) octaveHeight *= -1.0f;

        // Add height contribution of this octave to the final height value.
        finalHeight += octaveHeight * octaveWeight;

        // Update multipliers and offsets.
        densityMultiplier *= 2;
        octaveWeight *= falloff;
        gradientsOffset += octaveGradients;
    }

    // Store (not-normalized) height on map buffer.
    map[mapY * mapSize + mapX] = finalHeight;
}

/* Kernel which calculates the [min, max] range in which height values fall into. */
__global__ void computeHeightRange(float *map, float *heightRange, int mapSize) {
    // Define shared buffers
    __shared__ float blockMin[BLOCK_SIZE_1D];
    __shared__ float blockMax[BLOCK_SIZE_1D];

    // Check sanity.
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= mapSize * mapSize) {
        // Fill shared buffers with dummy values; then return.
        // NOTE: Unfortunally, some divergence is encountered, but it seems to be
        //       the only way to initialize values on the shared memory which are
        //       not relative to a pixel on the noise map.
        blockMin[threadIdx.x] = FLT_MAX;
        blockMax[threadIdx.x] = -FLT_MAX;
        return;
    }

    // Fill shared buffers with height values.
    blockMin[threadIdx.x] = map[linearIdx];
    blockMax[threadIdx.x] = map[linearIdx];
    __syncthreads();
    
    // Define the [min, max] range in which a height value can reside.
    // NOTE: This range will be later used in the normalization kernel to convert height
    //       values from [min, max] to [0, 255].
    // NOTE: At first, shared memory is used to calculate the range at a block level.
    //       Then, the actual range gets defined at a grid level using device memory.
    //       This way, the number of accesses to the device memory decreases.
    // NOTE: The reason why such range conversion doesn't occur in this kernel is that it
    //       whould imply a inter-block synchronization: before proceeding with the normalization,
    //       all blocks must have executed the last two atomic operations. Such synchronization
    //       can be achieved through cooperative groups, but that whould make the code incompatible
    //       with a considerable series of GPU architectures.
    
    // Parallel reduction on shared memory.
    for (int stride = blockDim.x / 2; stride > STRIDE_LIMIT; stride >>= 1) {
        if (threadIdx.x < stride)
            blockMin[threadIdx.x] = min(blockMin[threadIdx.x], blockMin[threadIdx.x + stride]);
        if (threadIdx.x < stride)
            blockMax[threadIdx.x] = max(blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);
        __syncthreads();
    }

    // Warp unrolling
#if WARP_UNROLLING
    if (threadIdx.x < 32) {
        volatile float *vMin = blockMin;
        volatile float *vMax = blockMax;
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 32]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 16]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 8]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 4]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 2]);
        vMin[threadIdx.x] = min(vMin[threadIdx.x], vMin[threadIdx.x + 1]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 32]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 16]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 8]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 4]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 2]);
        vMax[threadIdx.x] = max(vMax[threadIdx.x], vMax[threadIdx.x + 1]);
    }
#endif

    // Perform atomic operations to get the grid min and max.
    if (threadIdx.x == 0) atomicMinFloat(&heightRange[0], blockMin[0]); // Grid level min
    if (threadIdx.x == 0) atomicMaxFloat(&heightRange[1], blockMax[0]); // Grid level max
}

/* Kernel which normalizes the height values of a noise map within [min, max] to [0, 255], storing them in the char buffer. */
__global__ void normalizeMap(float *inMap, unsigned char *outMap, float *heightRange, int mapSize) {
    // Check sanity (is this pixel inside the map?).
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= mapSize * mapSize) return;

    // Retrieve height and [min, max] range.
    float height = inMap[linearIdx];
    float min = heightRange[0];
    float max = heightRange[1];

    // Convert result from [min, max] to [0, 255] range.
    // NOTE: How to convert a value from a range to another:
    //           scale = (newEnd - newStart) / (originalEnd - originalStart);
    //           value = (newStart + ((value - originalStart) * scale));
    float scale = 255.0f / (max - min);
    height = (height - min) * scale;
    if (height < 0.0f)   height = 0.0f;
    if (height > 255.0f) height = 255.0f;

    // Write result on (char) map.
    outMap[linearIdx] = (unsigned char)height;
}

/* Main host function. */
int main(int argc, char* argv[]) {   
    // Perlin noise parameters.
    char* outputPath = DEFAULT_OUTPUT_PATH;
    int mapSize = DEFAULT_MAP_SIZE;               // Size of the generated noise image.
    int noiseDensity = DEFAULT_NOISE_DENSITY;     // Frequency of the gradients' grid.
    int desiredOctaves = DEFAULT_DESIRED_OCTAVES; // Desired number of octaves to compute.
    float falloff = DEFAULT_FALLOFF;              // Falloff of an octave's weight, incrementally applied at each octave's weight.
    bool absoluteValue = DEFAULT_ABSOLUTE_VALUE;  // Whether to store the absolute values of computed heights.

    // Retrieve arguments.
    bool gotErrors = false;
    if (argc > 1) {
        outputPath = argv[1];
        char* end;
        int tempInt;
        float tempFloat;
        if (argc > 2) {
            errno = 0;
            tempInt = (int)strtol(argv[2], &end, 0);
            if (errno == 0 && *end == '\0' && tempInt > 0) mapSize = tempInt;
            else {
                fprintf(stderr, "Error: invalid MapSize (%s); using default.\n", argv[2]);
                gotErrors = true;
            }
            if (argc > 3) {
                errno = 0;
                tempInt = (int)strtol(argv[3], &end, 0);
                if (errno == 0 && *end == '\0' && tempInt > 1 && tempInt <= mapSize) noiseDensity = tempInt;
                else {
                    fprintf(stderr, "Error: invalid NoiseDensity (%s); using default.\n", argv[3]);
                    gotErrors = true;
                }
                if (argc > 4) {
                    errno = 0;
                    tempInt = (int)strtol(argv[4], &end, 0);
                    if (errno == 0 && *end == '\0' && tempInt > 0) desiredOctaves = tempInt;
                    else {
                        fprintf(stderr, "Error: invalid DesiredOctaves (%s); using default.\n", argv[4]);
                        gotErrors = true;
                    }
                    if (argc > 5) {
                        errno = 0;
                        tempFloat = atof(argv[5]);
                        if (errno == 0 && *end == '\0') falloff = tempFloat;
                        else {
                            fprintf(stderr, "Error: invalid Falloff (%s); using default.\n", argv[5]);
                            gotErrors = true;
                        }
                        if (argc > 6) {
                            if (strcmp(argv[6], "true") == 0) absoluteValue = true;
                            else if (strcmp(argv[6], "false") == 0) absoluteValue = false;
                            else {
                                fprintf(stderr, "Error: invalid AbsoluteValue (%s); using default.\n", argv[6]);
                                gotErrors = true;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        printf("-------------------------------------------------------------------------------------------------\n");
        printf("CUDA Perlin Noise Generator; Luigi Rapetta, 2022.\n");
        printf("Usage: perlinGenerator OutputPath? MapSize? NoiseDensity? DesiredOctaves? Falloff? AbsoluteValue?\n");
        printf("-------------------------------------------------------------------------------------------------\n");
        printf("[string]     OutputPath:     path of the output PNG image; default is \"%s\".\n", DEFAULT_OUTPUT_PATH);
        printf("[uint]       MapSize:        size of the generated noise image; default is %d.\n", DEFAULT_MAP_SIZE);
        printf("[uint]       NoiseDensity:   frequency of the gradients' grid; it's less or equal than MapSize; default is %d.\n", DEFAULT_NOISE_DENSITY);
        printf("[uint]       DesiredOctaves: desired number of octaves to compute; default is %d.\n", DEFAULT_DESIRED_OCTAVES);
        printf("[float]      Falloff:        falloff of an octave's weight, incrementally applied at each octave's weight; default is %.1f.\n", DEFAULT_FALLOFF);
        printf("[true/false] AbsoluteValue:  whether to store the absolute values of computed heights; default is %s.\n", DEFAULT_ABSOLUTE_VALUE ? "true" : "false");
    }

#if DEBUG
    if (argc <= 1 || gotErrors) printf("\n");
    printf("----------\n");
    printf("Parameters\n");
    printf("----------\n");
    printf("OutputPath:     \"%s\"\n", outputPath);
    printf("MapSize:        %d\n", mapSize);
    printf("NoiseDensity:   %d\n", noiseDensity);
    printf("DesiredOctaves: %d\n", desiredOctaves);
    printf("Falloff:        %f\n", falloff);
    printf("AbsoluteValue:  %s\n\n", absoluteValue ? "true" : "false");
#endif

    // Create events for execution time calculation.
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    // Determine the actual number of octaves and gradients.
    // NOTE: The density of a gradients' grid has to be less or equal than the map size.
    //       For each octave, such density doubles. Hence, the actual number of octaves may
    //       be less than the desired one if one or more densities break the aforementioned
    //       constraint.
    int octavesNumber = 0;
    int gradientsNumber = 0;
    int densityMultiplier = 1;
    for (int i = 0; i < desiredOctaves; i++) {
        int density = noiseDensity * densityMultiplier;
        if (density <= mapSize) {
            // The i-th octave is valid.
            octavesNumber++;

            // Keep track of the total number of gradients.
            gradientsNumber += (density + 1) * (density + 1);

            // Double the density.
            densityMultiplier *= 2;
        }
        else i = desiredOctaves; // Break cycle.
    }

#if DEBUG
    printf("-------------\n");
    printf("Octaves' data\n");
    printf("-------------\n");
    printf("Octaves number:      %d\n", octavesNumber);
    printf("Total gradients:     %d\n\n", gradientsNumber);
#endif

    // Allcoate memory for the noise map.
    // NOTE: The final map will be stored in the char matrix, while the float
    //       one serves as a working buffer. The char matrix will be filled at
    //       the end of the normalization process.
    unsigned char *h_map, *d_map;
    float *d_floatMap;
    CHECK(cudaMalloc(&d_map, sizeof(unsigned char) * mapSize * mapSize));
    CHECK(cudaMalloc(&d_floatMap, sizeof(float) * mapSize * mapSize));
    CHECK(cudaMemset(d_floatMap, 0, sizeof(float) * mapSize * mapSize)); // Set values of float matrix to 0.
#if PINNED_MEMORY
    CHECK(cudaMallocHost(&h_map, sizeof(unsigned char) * mapSize * mapSize));
#else
    h_map = (unsigned char*)malloc(sizeof(unsigned char) * mapSize * mapSize);
#endif
#if DEBUG
    printf("---------------------\n");
    printf("Map memory allocation\n");
    printf("---------------------\n");
    printf("Bytes for char matrix:  %d\n", sizeof(unsigned char) * mapSize * mapSize);
    printf("Bytes for float matrix: %d\n", sizeof(float) * mapSize * mapSize);
    printf("Total bytes allocated:  %d\n\n", sizeof(unsigned char) * mapSize * mapSize + sizeof(float) * mapSize * mapSize);
#endif

    // Allcoate memory for the gradients.
    // NOTE: A gradient is a 2D vector, hence it's represented by two floats.
    float *d_gradients;
    CHECK(cudaMalloc(&d_gradients, sizeof(float) * 2 * gradientsNumber));
#if DEBUG
    printf("---------------------------\n");
    printf("Gradients memory allocation\n");
    printf("---------------------------\n");
    printf("Gradients number: %d\n", gradientsNumber);
    printf("Bytes allocated:  %d\n\n", sizeof(float) * 2 * gradientsNumber);
#endif

    // Allcoate memory for the height range buffer.
    // NOTE: This buffer will be later filled int the computeHeightRange kernel in
    //       order to store the [min, max] range in which height values fall into.
    float *d_heightRange;
    CHECK(cudaMalloc(&d_heightRange, sizeof(float) * 2));
    CHECK(cudaMemset(d_heightRange,     0, sizeof(float))); // Initialize min value.
    CHECK(cudaMemset(d_heightRange + 1, 0, sizeof(float))); // Initialize max value.

    // Calculate all the gradients on multiple streams.
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * STREAM_NUMBER);
    int gradientOffset = 0;
    int slabSize = gradientsNumber / STREAM_NUMBER;
    int gridSizeGradients = (slabSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    srand((unsigned)time(NULL));
    for (int i = 0; i < STREAM_NUMBER; i++) {
        // Create the stream.
        CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));

        // If it's the last stream, consder the remainder to calculate the grid dimension.
        if (i == STREAM_NUMBER - 1) {
            int reminder = gradientsNumber % STREAM_NUMBER;
            gridSizeGradients = (slabSize + reminder + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
            slabSize += reminder;
        }

        // Generate gradients on this stream.
        generateGradients<<<gridSizeGradients, BLOCK_SIZE_1D, 0, stream[i]>>>(d_gradients + gradientOffset, slabSize, gradientsNumber, rand());

        // Update offset.
        gradientOffset += slabSize;
    }
    CHECK(cudaDeviceSynchronize());

    // Generate Perlin noise.
    slabSize = mapSize / STREAM_NUMBER;
    int slabOffset = 0;
    int gridSizePerlinX = (mapSize + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
    int gridSizePerlinY = (slabSize + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
    dim3 gridDimPerlin(gridSizePerlinX, gridSizePerlinY, 1);
    dim3 blockDimPerlin(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
    for (int i = 0; i < STREAM_NUMBER; i++) {
        // If it's the last stream, consder the remainder to calculate the grid dimension.
        if (i == STREAM_NUMBER - 1) {
            int reminder = mapSize % STREAM_NUMBER;
            gridDimPerlin.y = (slabSize + reminder + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            slabSize += reminder;
        }
        
        // Run noise generation kernel.
        perlinNoise<<<gridDimPerlin, blockDimPerlin, 0, stream[i]>>>(d_floatMap, d_gradients, mapSize, slabSize, slabOffset, noiseDensity, octavesNumber, gradientsNumber, falloff, absoluteValue);
    
        // Update offset.
        slabOffset += slabSize;
    }
    CHECK(cudaDeviceSynchronize());

    // Normalize the map in the range [0, 255].
    int gridSizeNormalize = (mapSize * mapSize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    computeHeightRange<<<gridSizeNormalize, BLOCK_SIZE_1D>>>(d_floatMap, d_heightRange, mapSize);
    normalizeMap<<<gridSizeNormalize, BLOCK_SIZE_1D>>>(d_floatMap, d_map, d_heightRange, mapSize);

    // Copy resulting noise map to host.
    CHECK(cudaMemcpy(h_map, d_map, sizeof(unsigned char) * mapSize * mapSize, cudaMemcpyDeviceToHost));

    // Store map into a PNG image file.
    if (stbi_write_png(outputPath, mapSize, mapSize, 1, h_map, mapSize) == 0) {
        fprintf(stderr, "Error: could not write on path \"%s\".\n", outputPath);
        printf("Writing on default path \"%s\".\n", DEFAULT_OUTPUT_PATH);
        if (stbi_write_png(DEFAULT_OUTPUT_PATH, mapSize, mapSize, 1, h_map, mapSize) == 0) {
            fprintf(stderr, "Error: could not write on default path \"%s\".\n", DEFAULT_OUTPUT_PATH);
        }
    } else printf("Noise map successfully written at \"%s\".\n",outputPath);

    // Print execution time.
    float elapsedTime;
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Device execution time: %fms.\n", elapsedTime);

    // Destroy streams.
    for (int i = 0; i < STREAM_NUMBER; i++)
        CHECK(cudaStreamDestroy(stream[i]));
    
    // Free memory.
#if PINNED_MEMORY
    CHECK(cudaFreeHost(h_map));
#else
    free(h_map);
#endif
    CHECK(cudaFree(d_map));
    CHECK(cudaFree(d_floatMap));
    CHECK(cudaFree(d_gradients));
    CHECK(cudaFree(d_heightRange));

    // Execute Perlin noise on host and compare execution times.
#if DEBUG
#if DEBUG_COMPARE
    // Create events for the Perlin noise execution on host.
    cudaEvent_t hostStart;
    cudaEvent_t hostStop;
    CHECK(cudaEventCreate(&hostStart));
    CHECK(cudaEventCreate(&hostStop));
    
    // Execute on host.
    CHECK(cudaEventRecord(hostStart));
    printf("\nExecuting on host...");
    h_perlinNoise(outputPath, mapSize, noiseDensity, desiredOctaves, falloff, absoluteValue);
    printf(" done!\n");
    CHECK(cudaEventRecord(hostStop));
    CHECK(cudaEventSynchronize(hostStop));

    // Print time and comparisons.
    float hostElapsedTime;
    CHECK(cudaEventElapsedTime(&hostElapsedTime, hostStart, hostStop));
    printf("Host execution time: %fms.\n", hostElapsedTime);
    printf("Speedup: %f.\n", hostElapsedTime / elapsedTime);
#endif // DEBUG_COMPARE
#endif // DEBUG

    // Return.
    return 0;
}