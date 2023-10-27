main: gpulz.cu
	nvcc -Xcompiler -fPIC -shared -o gpulz.so gpulz.cu

clean:
	rm -rf gpulz.so