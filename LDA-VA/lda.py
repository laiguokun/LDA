from readfile import readfile;
import numpy as np;
import scipy.special as sp;
import math;
import time;
import random;

alpha = 0;
beta = [];
K = 10;
Gamma = {};
Phi = {};
doc = {};
doc_size = 0;
voca_size = 0;
timer = 0;

def init():
	global alpha,beta,Gamma,Phi,doc;

	#random start
	beta = np.ones((K,voca_size));
	beta = beta/K;
	for v in range(voca_size):
		tot = 0;
		for i in range(K):
			beta[i][v] += random.random()/10;
			tot += beta[i][v];
		for i in range(K):
			beta[i][v] = beta[i][v] / tot;


	Gamma = [];
	Phi = [];
	for d in range(doc_size):
		Phi.append(np.ones((len(doc[d]), K))/K);
		Gamma.append(np.ones(K));
	alpha = 1;

def compute_mle(d):
	global alpha,beta,Gamma,Phi,doc,timer;
	res = 0;
	res += sp.gammaln(K * alpha);
	res -= K * sp.gammaln(alpha);
	gamma_sum = np.sum(Gamma[d]);
	length = len(doc[d]);
	psi = [];
	now = time.time();
	
	for i in range(K):
		psi.append(sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));

	for i in range(K):
		res += (alpha - 1) * psi[i];
		res += sp.gammaln(Gamma[d][i]);
		res -= (Gamma[d][i] - 1) * psi[i];

	for n in range(length):
		for i in range(K):
			res += Phi[d][n][i] * psi[i];
			res -= Phi[d][n][i] * math.log(Phi[d][n][i]);
			res += Phi[d][n][i] * math.log(beta[i][doc[d][n]]);
	res -= sp.gammaln(gamma_sum);
	timer += time.time() - now;
	return res;

def compute_alpha_mle(alpha):
	res = 0;
	const = 0;
	for d in range(doc_size):
		res += sp.gammaln(K * alpha);
		res -= K * sp.gammaln(alpha);
		gamma_sum = np.sum(Gamma[d]);
		for i in range(K):
			const += (sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));	
	res += (alpha - 1) * const;
	return res, const; 


def mle():
	global alpha,beta,Gamma,Phi,doc,timer;
	timer = 0;
	res = 0;
	for d in range(doc_size):
		res += compute_mle(d);
	print(timer);
	return res;

def Estep(d, max_iter):
	global alpha,beta,Gamma,Phi,doc;
	Phi[d] = np.ones((doc_size, K)) / K;
	Gamma[d] = np.ones(K) * len(doc[d]) / K + alpha;
	last = 0;
	now = compute_mle(d);
	iter_num = 0;
	while(abs(last - now) > 1e-9 and iter_num < max_iter):

		gamma_sum = sum(Gamma[d]);
		for n in range(len(doc[d])):
			for i in range(K):
				Phi[d][n][i] = beta[i][doc[d][n]] * math.exp(sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));
			Phi[d][n] = Phi[d][n] / np.sum(Phi[d][n]);
		for i in range(K):
			Gamma[d][i] = 0;
			for n in range(len(doc[d])):
				Gamma[d][i] += Phi[d][n][i];
		Gamma[d] += alpha;
		last = now;
		now = compute_mle(d);
		iter_num += 1;

def backtrack(alpha,dx,da):
	t = 1;
	while (compute_alpha_mle(alpha + t * dx) > compute_alpha_mle(alpha) + alpha * t * da * dx):
		t = t * beta;
	alpha = alpha + t * dx;
	return alpha;

	
def Mstep(max_iter):
	global alpha,beta,Gamma,Phi,doc;
	#update beta
	beta = np.zeros((K,voca_size));
	for i in range(K):
		for d in range(doc_size):
			for n in range(len(doc[d])):
				beta[i][doc[d][n]] += Phi[d][n][i];
	beta_sum = np.sum(beta, axis = 0);
	for i in range(voca_size):
		beta[:,i] = beta[:,i]/beta_sum[i];
	#update alpha
	last = 0;
	const = 0;
	now, const = compute_alpha_mle(alpha);
	while (abs(last - now) > 1e-9 and iter_num < max_iter):
		da = K * (doc_size * (sp.psi(alpha * K) - sp.psi(alpha))) + const;
		dda = K * (doc_size * (K * sp.polygamma(1, alpha * K) - sp.polygamma(1, alpha)));
		dx = -da/dda;
		alpha = backtrack(alpha,dx,da);
		last = now;
		now, const = compute_alpha_mle(alpha);
		iter_num += 1;


def savemodel():
	global alpha,beta,Gamma,Phi,doc;
	fout = open('lda_model.txt', 'w');
	fout.write(str(alpha) + '\n');
	for i in range(K):
		for v in range(voca_size):
			fout.write(beta[i][v] + ' ');
		fout.write('\n');



def train(max_iter):
	global alpha,beta,Gamma,Phi,doc;
	for i in range(max_iter):
		now = mle();
		print(now);
		print('Estep');
		for d in range(doc_size):
			Estep(d, 20); # the e step of em algorithm
			if (d % 100 == 0):
				print('*', end = "");
		now = mle();
		print(now);
		print('Mstep');
		Mstep(20); # the m step of em algorith
	savemodel();


if __name__ == '__main__':
	fin = readfile('ap.dat');
	doc, voca= fin.read();
	voca_size = len(voca);
	doc_size = len(doc);
	init();
	train(10);