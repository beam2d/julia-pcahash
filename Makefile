hamming_distance.so: hamming_distance.c
	gcc -std=c99 -O2 -msse4.2 -fPIC -shared $^ -o $@
