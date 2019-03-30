#!/usr/bin/python3

# Author: Deepak Pandita
# Date created: 30 Mar 2018

import numpy as np
from scipy.stats import multivariate_normal
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#This function computes the log-likelihood
def loglikelihood(data, lambda_k, mu, cov):
	ll = 0.0
	for index in range(len(data)):
		p = 0.0
		for k in range(len(mu)):
			p += lambda_k[k] * multivariate_normal.pdf(data[index], mu[k], cov[k])
		ll += np.log(p)
	return ll

#Loglikelihood for tied covariance matrix
def loglikelihood_tied(data, lambda_k, mu, cov):
	ll = 0.0
	for index in range(len(data)):
		p = 0.0
		for k in range(len(mu)):
			p += lambda_k[k] * multivariate_normal.pdf(data[index], mu[k], cov)
		ll += np.log(p)
	return ll

def main():
	#using optional parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--clusters', action="store", help = "Initial no. of clusters", type = int)
	parser.add_argument('--max_iter', action="store", help = "Maximum no. of iterations", type = int)
	parser.add_argument('--tied', help = "Use tied covariance matrix", action = "store_true")
	args = parser.parse_args()

	#file paths
	data_file = 'points.dat'

	#default no. of clusters and iterations
	clusters = 4
	max_iter = 15
	tied = False

	if args.clusters:
		clusters = args.clusters
	if args.max_iter:
		max_iter = args.max_iter
	if args.tied:
		tied = True

	print("Clusters: " + str(clusters))

	#Read data file
	print('Reading data file: '+data_file)
	f = open(data_file)
	lines = f.readlines()
	f.close()

	list = []
	for line in lines:
		temp_list = []
		temp_list.append(float(line.split()[0]))
		temp_list.append(float(line.split()[1]))
		list.append(temp_list)
	
	data = np.array(list)
	

	#Split the dataset into train and dev
	split_size = int(len(list)*0.9)
	train, dev = data[:split_size], data[split_size:]

	print("Dimensions of Train: ",train.shape)
	


	#Initialize mean and covariance
	rand = np.random.randint(0,len(train),size=clusters)
	
	err=[]
	cov=[]
	if tied:
		cov = np.cov(train.T)
		print(cov.shape)
		err = np.random.multivariate_normal([0,0],cov,size=clusters)
	else:
		cov = [np.cov(train.T) for i in range(clusters)]
		err = np.random.multivariate_normal([0,0],cov[0],size=clusters)
	mu = train[rand]+err
	
	lambda_k = [1.0/clusters for i in range(clusters)]
	
	
	#Plotting the data
	plt.figure(1)
	plt.plot(train[:,0],train[:,1], 'ro')
	plt.plot(mu[:,0],mu[:,1],'mo',label='Initial means')
	
	
	train_ll=[]
	dev_ll=[]
	old_ll_train = 0.0
	old_ll_dev = 0.0
	#Calculate initial log-likelihood
	if tied:
		old_ll_train = loglikelihood_tied(train,lambda_k,mu,cov)
		old_ll_dev = loglikelihood_tied(dev,lambda_k,mu,cov)
	else:
		old_ll_train = loglikelihood(train,lambda_k,mu,cov)
		old_ll_dev = loglikelihood(dev,lambda_k,mu,cov)
	train_ll.append(old_ll_train)
	dev_ll.append(old_ll_dev)

	#Start EM algorithm
	for iter in range(max_iter):
		if iter%5==0:
			print('Iteration',iter)
		
		mat = np.zeros((len(train), clusters))
		mu_prime = np.zeros((clusters,train.shape[1]))
		cov_prime = []
		if tied:
			cov_prime = np.zeros((train.shape[1],train.shape[1]))
		else:
			cov_prime = [np.zeros((train.shape[1],train.shape[1])) for i in range(clusters)]
		

		#E-Step
		for index in range(len(train)):
			for k in range(clusters):
				if tied:
					mat[index, k] = lambda_k[k] * multivariate_normal.pdf(train[index], mu[k], cov)
				else:
					mat[index, k] = lambda_k[k] * multivariate_normal.pdf(train[index], mu[k], cov[k])
		row_sums = mat.sum(axis=1)[:, np.newaxis]
		mat = mat / row_sums
		

		#M-Step
		sum_zn = np.sum(mat, axis=0)
		cov_tied = np.zeros((train.shape[1],train.shape[1]))
		for k in range(clusters):
			sum_zx = [0.0,0.0]
			for i in range(len(train)):
				sum_zx += (mat[i,k]*train[i])
			
			#Update lambda
			lambda_k[k] = sum_zn[k]/len(train)
			
			#Update mu
			mu[k] = sum_zx/sum_zn[k]
			
			#Update cov
			if tied:
				for i in range(len(train)):
					cov_tied += lambda_k[k]*np.outer(train[i] - mu[k], train[i] - mu[k])
			else:
				sum_cov_k = np.zeros((train.shape[1],train.shape[1]))
				for i in range(len(train)):
					sum_cov_k += mat[i,k]*np.outer(train[i] - mu[k], train[i] - mu[k])
				cov[k] = sum_cov_k/sum_zn[k]
		new_ll_train=0.0
		new_ll_dev=0.0
		
		if tied:
			cov = cov_tied/len(train)
			#Calculate log-likelihood
			new_ll_train = loglikelihood_tied(train,lambda_k,mu,cov)
			new_ll_dev = loglikelihood_tied(dev,lambda_k,mu,cov)
		else:
			new_ll_train = loglikelihood(train,lambda_k,mu,cov)
			new_ll_dev = loglikelihood(dev,lambda_k,mu,cov)
		
		train_ll.append(new_ll_train)
		dev_ll.append(new_ll_dev)
		
		#If the increase in likelihood is less than 1e-6 then stop
		if (new_ll_dev - old_ll_dev) < 1e-6 and new_ll_dev > -np.inf:
			break
		
		old_ll_train = new_ll_train
		old_ll_dev = new_ll_dev
	print('Log-Likelihoods on train:',train_ll)
	print('Log-Likelihoods on dev:',dev_ll)
	
	#Plot final means
	plt.plot(mu[:,0],mu[:,1],'bo',label='Final means')
	plt.title("Training Data")
	plt.legend()
	
	plt.figure(2)
	plt.plot(dev[:,0],dev[:,1], 'ro')
	plt.plot(mu[:,0],mu[:,1],'bo',label='Final means')
	plt.title("Dev Data")
	plt.legend()
	
	#Plotting log-likelihood
	plt.figure(3)
	plt.plot(train_ll, label='Train')
	plt.xlabel('Iterations')
	plt.ylabel('LogLikelihood')
	plt.title('Log-Likelihood on Train')
	plt.legend()
	
	plt.figure(4)
	plt.plot(dev_ll, label='Dev')
	plt.xlabel('Iterations')
	plt.ylabel('LogLikelihood')
	plt.title('Log-Likelihood on Dev')
	plt.legend()
	plt.show()
if __name__ == '__main__':
	main()