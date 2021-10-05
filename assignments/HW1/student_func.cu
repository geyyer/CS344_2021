// Homework 1
// Color to Greyscale Conversion

// A common way to represent color images is known as RGBA - the color
// is specified by how much Red, Grean and Blue is in it.
// The 'A' stands for Alpha and is used for transparency, it will be
// ignored in this homework.

// Each channel Red, Blue, Green and Alpha is represented by one byte.
// Since we are using one byte for each color there are 256 different
// possible values for each color.  This means we use 4 bytes per pixel.

// Greyscale images are represented by a single intensity value per pixel
// which is one byte in size.

// To convert an image from color to grayscale one simple method is to
// set the intensity to the average of the RGB channels.  But we will
// use a more sophisticated method that takes into account how the eye
// perceives color and weights the channels unequally.

// The eye responds most strongly to green followed by red and then blue.
// The NTSC (National Television System Committee) recommends the following
// formula for color to greyscale conversion:

// I = .299f * R + .587f * G + .114f * B

// Notice the trailing f's on the numbers which indicate that they are
// single precision floating point constants and not double precision
// constants.

// You should fill in the kernel as well as set the block and grid sizes
// so that the entire image is processed.

#include "utils.h"

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage, 
                                  int numRows,
                                  int numCols) {

  unsigned char r, g, b;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    idx < numRows * numCols; 
    idx += gridDim.x * blockDim.x) {
    r = rgbaImage[idx].x;
    g = rgbaImage[idx].y;
    b = rgbaImage[idx].z;
    greyImage[idx] = .299f * r + .587f * g + .114f * b;
  }
}

void your_rgba_to_greyscale(const uchar4 *const h_rgbaImage,
                            uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage, 
                            size_t numRows,
                            size_t numCols) {
  // You must fill in the correct sizes for the blockSize and gridSize
  // currently only one block with one thread is being launched
  const size_t pixels = numRows * numCols;
  const size_t thread_num = 256;
  const dim3 blockSize(thread_num, 1, 1);
  const dim3 gridSize((pixels + thread_num - 1) / thread_num, 1);
  printf("Starting execution \n");
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, 
                                             d_greyImage, 
                                             numRows,
                                             numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
