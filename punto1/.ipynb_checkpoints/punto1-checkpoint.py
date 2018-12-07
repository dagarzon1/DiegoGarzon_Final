import numpy as np
import matplotlib.pyplot as plt
import glob

k = sorted(glob.glob("*.txt"))
dat = []
V = []
N = 1000
means = np.zeros(8)
varis = np.zeros(8)

x = np.linspace(-4,4,1000)
s = 1.0
mu = 0.0
y = 1.0/( s * np.sqrt(2 * np.pi) ) * np.exp( - ( x - mu ) * ( x - mu) / ( 2.0 * s * s ) )

plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
for i in k:
	d = np.loadtxt(i)
	plt.hist(d, alpha=0.3, density=True)
plt.plot(x,y)
plt.ylabel('freq')
plt.xlabel('x')


for i in range(2,1000):
	m = 0
	for j in k:
		d = np.loadtxt(j)
		means[m] = np.mean(d[:i])
		varis[m] = np.std(d[:i]) ** 2.0
		m+=1
	mean_dat = np.mean(means)
	means = means - mean_dat
	means = means ** 2.0
	b = i / (8-1) * np.sum(means)
	w = 1/8 * np.sum(varis)
	v = (i-1) * w / i + ( (8+1) * b / (8*i) )
	V.append(v)

plt.subplot(1,2,2)
plt.plot(np.arange(2,1000),V)
plt.ylabel('Gelman-Rubin variable')
plt.xlabel('Number of iterations')

plt.savefig('punto1.png')