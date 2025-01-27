g++ -shared -fPIC -o libtinytorch_cpu.so tinytorch.cpp && python user_side.py
nvcc -Xcompiler -fPIC -shared -o libtinytorch_cuda.so tinytorch.cu && DEVICE=CUDA python user_side.py 
