main: gpulz.cu
	nvcc -O3 -Xcompiler -fPIC -shared gpulz.cu -arch sm_86 -o gpulz.so

clean:
	rm -rf gpulz