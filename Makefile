main: gpulz.cu
	nvcc -Xcompiler -fPIC -shared -lcufile -o gpulz.so gpulz.cu

clean:
	rm -rf gpulz.so