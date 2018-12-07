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

j=0
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
for i in k:
	d = np.loadtxt(i)
	plt.hist(d, alpha=0.3, density=True)
	means[j] = np.mean(d)
	varis[j] = np.std(d) ** 2.0
	j +=1
plt.plot(x,y)
plt.ylabel('freq')
plt.xlabel('x')

mean_dat = np.mean(means)
means = means - mean_dat
means = means ** 2.0
for i in range(2,8):
	b = N / (i-1) * np.sum(means[:i])
	w = 1/i * np.sum(varis[:i])
	v = (N-1) * w / N + ( (i+1) * b / (i*N) )
	V.append(v)

plt.subplot(1,2,2)
plt.plot(np.arange(2,8),V)
plt.ylabel('Gelman-Rubin variable')
plt.xlabel('Number of chains')

plt.savefig('punto1.png')