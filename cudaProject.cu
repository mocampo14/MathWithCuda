#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Lets you use the Cuda FFT library
#include "cufft.h"

#include <stdio.h>
#include "cudaProject.h"


cudaError_t mathWithCuda(float *output, float *input1, float *input2, unsigned int size, int oper);

// Using __global__ to declare function as device code (GPU)
// Do the math inside here:
__global__ void mathKernel(float *output, float *input1, float *input2, int n, int oper)
{
	// Allocate elements to threads
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Avoid access beyond the end of the array
	if (i < n)
	{
		// No for-loop needed, CUDA runtime will thread this
		switch (oper)
		{
		case 1: // Addition
			output[i] = input1[i] + input2[i];
			break;
		case 2: // Subtraction
			output[i] = input1[i] - input2[i];
			break;
		case 3: // Multiplication
			output[i] = input1[i] * input2[i];
			break;
		case 4: // Division
			output[i] = input1[i] / input2[i];
			break;

			// Add more operations here:
		case 5:
			break;
		case 6:
			break;
		case 7:
			break;

		default:
			return;
		}

		// Ensure all the data is available
		__syncthreads(); // Gives a syntax "error" but this doesn't give build errors
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mathWithCuda(float *output, float *input1, float *input2, unsigned int size, int oper)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;

	bool error = false;
	while (1)
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			error = true;
			break;
		}

		// Allocate GPU buffers for three vectors
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			error = true;
			break;
		}

		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			error = true;
			break;
		}

		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			error = true;
			break;
		}

		// Copy input vectors from host memory to device buffers ( cpu -> gpu )
		cudaStatus = cudaMemcpy(dev_a, input1, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			error = true;
			break;
		}

		cudaStatus = cudaMemcpy(dev_b, input2, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			error = true;
			break;
		}


		// Define timers
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Start counting GPU processing time
		cudaEventRecord(start);

		// Launch a kernel on the GPU with n thread for 1 element.
		mathKernel << <size, 1 >> >(dev_c, dev_a, dev_b, size, oper);

		// Stop counting GPU processing time
		cudaEventRecord(stop);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			error = true;
			break;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			error = true;
			break;
		}

		// Copy output vector from device buffer to host memory ( gpu -> cpu )
		cudaStatus = cudaMemcpy(output, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			error = true;
			break;
		}

		// Wait until everything is done -- not included in gpu timecount
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		float bandWidth = size / milliseconds;

		// Print time results
		printf("Total GPU time passed: %f ms \n", milliseconds);
		printf("Total GPU bandwidth: %f\n", bandWidth);

		break;
	}

	// Clean-up memory
	if (error)
	{
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	}
	return cudaStatus;
}

void cudaProject::mathVectors(float* c, float* a, float* b, int n, int oper)
{
	// Add vectors in parallel.
	// cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	cudaError_t cudaStatus = mathWithCuda(c, a, b, n, oper);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaProject::sumVectors failed!");
		return;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}
}
