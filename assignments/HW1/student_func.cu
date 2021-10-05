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

struct __align__(32) uchar32 {
  uchar4 x;
  uchar4 y;
  uchar4 z;
  uchar4 w;
  uchar4 a;
  uchar4 b;
  uchar4 c;
  uchar4 d;
};

struct __align__(8) uchar8 {
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
  unsigned char a;
  unsigned char b;
  unsigned char c;
  unsigned char d;
};

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage, 
                                  int numRows,
                                  int numCols) {

  int pixels = numRows * numCols;
  int eighth = pixels / 8;
  int limit = eighth * 8;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < eighth) {
    uchar32 load = ((const uchar32*) rgbaImage)[idx];
    uchar4 l0 = load.x;
    float tile0 = .299f * l0.x + .587f * l0.y + .114f * l0.z;
    uchar4 l1 = load.y;
    float tile1 = .299f * l1.x + .587f * l1.y + .114f * l1.z;
    uchar4 l2 = load.z;
    float tile2 = .299f * l2.x + .587f * l2.y + .114f * l2.z;
    uchar4 l3 = load.w;
    float tile3 = .299f * l3.x + .587f * l3.y + .114f * l3.z;
    uchar4 l4 = load.a;
    float tile4 = .299f * l4.x + .587f * l4.y + .114f * l4.z;
    uchar4 l5 = load.b;
    float tile5 = .299f * l5.x + .587f * l5.y + .114f * l5.z;
    uchar4 l6 = load.c;
    float tile6 = .299f * l6.x + .587f * l6.y + .114f * l6.z;
    uchar4 l7 = load.d;
    float tile7 = .299f * l7.x + .587f * l7.y + .114f * l7.z;

    uchar8 out;
    out.x = (unsigned char) tile0;
    out.y = (unsigned char) tile1;
    out.z = (unsigned char) tile2;
    out.w = (unsigned char) tile3;
    out.a = (unsigned char) tile4;
    out.b = (unsigned char) tile5;
    out.c = (unsigned char) tile6;
    out.d = (unsigned char) tile7;

    ((uchar8*) greyImage)[idx] = out;    
  }
  
  if (limit + idx < pixels) {
    uchar4 load = rgbaImage[limit + idx];
    float tile = .299f * load.x + .587f * load.y + .114f * load.z;
    greyImage[limit + idx] = (unsigned char) tile;
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
  const size_t thread_num = 64;
  const dim3 blockSize(thread_num, 1, 1);
  const dim3 gridSize((pixels + thread_num * 8 - 1) / (thread_num * 8), 1, 1);
  printf("Starting execution \n");
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, 
                                             d_greyImage, 
                                             numRows,
                                             numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
