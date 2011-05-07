
#include "../lib/cuPrintf.cu"

__global__ void testKernel(int val)
{
  cuPrintf("Value is: %d\n", val);
}

void foo() {

  cudaPrintfInit();
  testKernel<<< 2, 3 >>>(10);
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();

}

int main()
{
  foo();
  return 0;
}
