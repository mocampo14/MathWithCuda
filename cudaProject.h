#pragma once

class __declspec(dllexport) cudaProject
{
public:
	cudaProject() {}
	~cudaProject() {}
	void mathVectors(float* c, float* a, float* b, int n, int oper);
};