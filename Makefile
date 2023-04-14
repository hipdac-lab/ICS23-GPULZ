main: gpulz.cu
	nvcc gpulz.cu -arch sm_80 -o gpulz

clean:
	rm -rf gpulz