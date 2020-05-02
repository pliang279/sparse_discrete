import numpy as np
import matplotlib.pyplot as plt

def plot_loss(fname, c1, name):
	train_losses = []
	val_losses = []
	nnz_userT = []
	nnz_itemT = []
	with open(fname) as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if 'Training loss' in line:
				train_losses.append(float(line[line.index('Training loss')+14:]))
			if 'Val loss' in line:
				val_losses.append(float(line[line.index('Val loss')+9:]))

			if 'nnz in user T' in line:
				nnz_userT.append(int(line[line.index('nnz in user T')+15:]))
			if 'nnz in item T' in line:
				nnz_itemT.append(int(line[line.index('nnz in item T')+15:]))

	print (nnz_userT)
	print (nnz_itemT)

	# plt.ylim(0,100)
	# plt.xlim(0,10)
	plt.xlabel('epoch', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.plot(val_losses, color=c1, label=name)
	plt.legend(loc='upper right')
	# plt.fill_between(x, means-sds, means+sds, 
	# 	alpha=0.1, edgecolor=c1, facecolor=c1,
    # 	linewidth=1, antialiased=True)

plot_loss('baseline16.txt', 'blue', 'baseline')
#plot_loss('sparse100_0.05.txt', 'black', 'sparse a = 100, lambda = 0.05')
#plot_loss('sparse500_0.05.txt', 'green', 'spase a = 500, lambda = 0.05')
#plot_loss('sparse1000_0.05.txt', 'red', 'sparse a = 1000, lambda = 0.05')
plot_loss('sparse100_0.01.txt', 'yellow', 'spase a = 100, lambda = 0.01')
plot_loss('sparse200_0.01.txt', 'orange', 'spase a = 200, lambda = 0.01')
plot_loss('sparse500_0.01.txt', 'red', 'spase a = 500, lambda = 0.01')
plot_loss('sparse1000_0.01.txt', 'green', 'sparse a = 1000, lambda = 0.01')
plt.show()