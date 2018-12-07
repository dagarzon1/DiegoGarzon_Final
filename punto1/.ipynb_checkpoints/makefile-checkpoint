punto1.png:sample_1.txt
	python punto1.py
sample_1.txt: punto1.x
	./punto1.x
	rm punto1.x
punto1.x:
	c++ -fopenmp punto1.c -o punto1.x
