#include "stb_image_write.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _HOST_PERLIN_H
#define _HOST_PERLIN_H


/* Definition of a 2D vector. */
struct Vector2 {
    float x;
    float y;
};

/* It linearly interpolates between two values. */
float inline h_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/* It returns the actual number of octaves to compute. */
int h_getOctavesNumber(int mapSize, int noiseDensity, int desiredOctaves) {
    int octavesNumber = 0;
    int densityMultiplier = 1;
    int density;
    for (int i = 0; i < desiredOctaves; i++) {
        density = noiseDensity * densityMultiplier;
        if (density <= mapSize) {
            // The i-th octave is valid.
            octavesNumber++;

            // Double the density.
            densityMultiplier *= 2;
        }
        else i = desiredOctaves; // Break cycle.
    }
    return octavesNumber;
}

/* It generates an array of random gradients. */
Vector2 *h_generateGradients(int density) {
    // Allocate memory for the gradients.
    int gradientsNumber = (density + 1) * (density + 1);
    Vector2 *gradients = (Vector2*)malloc(sizeof(Vector2) * gradientsNumber);
    
    // Generate gradients.
    for (int i = 0; i < gradientsNumber; i++) {
        // NOTE: To generate a random float with rand():
        //           random = LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)))
        // Source: https://stackoverflow.com/questions/686353/random-float-number-generation
        gradients[i].x = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2.0f)));
        gradients[i].y = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2.0f)));
    }

    // Return gradients.
    return gradients;
}

/* It computes the noise for this octave, storing it on the buffer. */
void h_perlinOctave(float *map, int mapSize, int density, float weight, bool abs) {
    // Generate gradients for this octave.
    Vector2* gradient = h_generateGradients(density);

    // Cycle among map coordinates.
    int gradientsDistance = mapSize / density; // The distance between two gradients, in pixels.
    for (int mapY = 0; mapY < mapSize; mapY++)
        for (int mapX = 0; mapX < mapSize; mapX++) {
            // Calculate which cell of the gradient grid this pixel is in.
            int gridX = mapX / gradientsDistance;
            int gridY = mapY / gradientsDistance;

            // Retrieve the indices of the four gradients nearest to this pixel.
            int gradientGridSize = density + 1;
            int octaveGradients = gradientGridSize * gradientGridSize;
            int tlIdx = (gridY * gradientGridSize + gridX) % octaveGradients;
            int trIdx = (gridY * gradientGridSize + gridX + 1) % octaveGradients;
            int blIdx = ((gridY + 1) * gradientGridSize + gridX) % octaveGradients;
            int brIdx = ((gridY + 1) * gradientGridSize + gridX + 1) % octaveGradients;

            // Calculate distance between the beginning of the grid's cell (top left) and the pixel.
            int deltaX = mapX - gradientsDistance * gridX;
            int deltaY = mapY - gradientsDistance * gridY;

            // Calculate height values of the gradients.
            float tlDot = (float)deltaX * gradient[tlIdx].x + (float)deltaY * gradient[tlIdx].y;
            float trDot = (float)(deltaX - gradientsDistance) * gradient[trIdx].x + (float)deltaY * gradient[trIdx].y;
            float blDot = (float)deltaX * gradient[blIdx].x + (float)(deltaY - gradientsDistance) * gradient[blIdx].y;
            float brDot = (float)(deltaX - gradientsDistance) * gradient[brIdx].x + (float)(deltaY - gradientsDistance) * gradient[brIdx].y;

            // Calculate the horizontal and vertical weights used for interpolation.
            float horWeight = (float)deltaX / (float)gradientsDistance;
            horWeight = (horWeight * (horWeight * 6.0 - 15.0) + 10.0) * horWeight * horWeight * horWeight;
            float verWeight = (float)deltaY / (float)gradientsDistance;
            verWeight = (verWeight * (verWeight * 6.0 - 15.0) + 10.0) * verWeight * verWeight * verWeight;

            // Interpolate values in order to get the height for this pixel.
            // NOTE: Interpolate twice horizontally, once vertically.
            float top    = h_lerp(tlDot, trDot, horWeight);
            float bottom = h_lerp(blDot, brDot, horWeight);
            float height = h_lerp(top, bottom, verWeight);

            // If desired, apply abs on height value.
            if (abs && height < 0.0f) height *= -1.0f;

            // Add (not-normalized) height on map buffer.
            map[mapY * mapSize + mapX] += height * weight;
        }


    // Deallocate memory.
    delete(gradient);
}

/* It calculates the [min, max] range in which height values fall into. */
void h_computeHeightRange(float *map, float *min, float *max, int mapSize) {
    // Cycle among map coordinates.
    float minHeight = FLT_MAX;
    float maxHeight = -FLT_MAX;
    for (int mapY = 0; mapY < mapSize; mapY++)
        for (int mapX = 0; mapX < mapSize; mapX++) {
            float height = map[mapY * mapSize + mapX];
            if (height > maxHeight) maxHeight = height;
            else if (height < minHeight) minHeight = height;
        }
    
    // Store range.
    *min = minHeight;
    *max = maxHeight;
}

/* It normalizes the height values of a noise map within [min, max] to [0, 255], storing them in the char buffer. */
void h_normalizeMap(float *inMap, unsigned char *outMap, float min, float max, int mapSize) {
    // Cycle among map coordinates.
    for (int mapY = 0; mapY < mapSize; mapY++)
        for (int mapX = 0; mapX < mapSize; mapX++) {
            // Convert result from [min, max] to [0, 255] range.
            // NOTE: How to convert a value from a range to another:
            //           scale = (newEnd - newStart) / (originalEnd - originalStart);
            //           value = (newStart + ((value - originalStart) * scale));
            float height = inMap[mapY * mapSize + mapX];
            float scale = 255.0f / (max - min);
            height = (height - min) * scale;
            if (height < 0.0f)   height = 0.0f;
            if (height > 255.0f) height = 255.0f;

            // Write result on (char) map.
            outMap[mapY * mapSize + mapX] = (unsigned char)height;
        }
}

/* It generates Perlin noise and stores the result on the given path. */
void h_perlinNoise(char *outputPath, int mapSize, int noiseDensity, int desiredOctaves, float falloff, bool absoluteValue) {

    // Determine the actual number of octaves to compute.
    int octavesNumber = h_getOctavesNumber(mapSize, noiseDensity, desiredOctaves);

    // Allcoate memory for the noise map.
    float *map = (float*)malloc(sizeof(float) * mapSize * mapSize);
    unsigned char *outMap = (unsigned char*)malloc(sizeof(unsigned char) * mapSize * mapSize);
    map = (float*)memset((void*)map, 0, sizeof(float) * mapSize * mapSize);

    // Generate noise for each octave.
    srand((unsigned)time(NULL));
    float weight = 1.0f;
    int densityMultiplier = 1;
    for (int i = 0; i < octavesNumber; i++) {
        h_perlinOctave(map, mapSize, noiseDensity * densityMultiplier, weight, absoluteValue);
        weight *= falloff;
        densityMultiplier *= 2;
    }

    // Normalize map.
    float min;
    float max;
    h_computeHeightRange(map, &min, &max, mapSize);
    h_normalizeMap(map, outMap, min, max, mapSize);

    // Store map into a PNG image file.
    stbi_write_png(outputPath, mapSize, mapSize, 1, outMap, mapSize);

    // Deallocate memory.
    delete(map);
    delete(outMap);
}


#endif