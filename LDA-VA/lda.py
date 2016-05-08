from readfile import readfile;
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
doc_cnt = [];
doc_size = 0;
voca_size = 0;
timer = 0;

def sum_vector(a):
	res = 0;
	for i in range(len(a)):
		res += a[i];
	return res;

def sum_matrix(a, op):
	res = []
	n = len(a);
	m = len(a[0]);
	if (op == 0):
		for j in range(m):
			res.append(0);
			for i in range(n):
				res[j] += a[i][j];

	return res;

def init():
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	#random start
	beta = [];
	for i in range(K):
		beta.append([]);
		tot = 0;
		for j in range(voca_size):
			beta[i].append(1/K + random.random()/10);
			tot += beta[i][j];
		for j in range(voca_size):
			beta[i][j] /= tot;

	Phi = [];
	Gamma = [];
	for d in range(doc_size):
		Phi.append([]);
		for n in range(len(doc[d])):
			Phi[d].append([]);
			for i in range(K):
				Phi[d][n].append(1/K);
		Gamma.append([]);
		for i in range(K):
			Gamma[d].append(1/K + alpha);
	alpha = 1;

def compute_mle(d):
	global alpha,beta,Gamma,Phi,doc,timer,doc_cnt;
	res = 0;
	res += sp.gammaln(K * alpha);
	res -= K * sp.gammaln(alpha);
	gamma_sum = sum_vector(Gamma[d]);
	length = len(doc[d]);
	psi = [];

	for i in range(K):
		psi.append(sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));

	for i in range(K):
		res += (alpha - 1) * psi[i];
		res += sp.gammaln(Gamma[d][i]);
		res -= (Gamma[d][i] - 1) * psi[i];

	now = time.time();

	for n in range(length):
		for i in range(K):
			res += doc_cnt[d][n] * Phi[d][n][i] * psi[i];
#			res -= doc_cnt[d][n] * Phi[d][n][i] * math.log(Phi[d][n][i]);
			res += doc_cnt[d][n] * Phi[d][n][i] * math.log(beta[i][doc[d][n]]/Phi[d][n][i]);

	timer += time.time() - now;
	res -= sp.gammaln(gamma_sum);
	return res;

def compute_alpha_mle(alpha):
	global beta,Gamma,Phi,doc,timer,doc_cnt;
	res = 0;
	const = 0;
	for d in range(doc_size):
		res += sp.gammaln(K * alpha);
		res -= K * sp.gammaln(alpha);
		gamma_sum = sum_vector(Gamma[d]);
		for i in range(K):
			const += (sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));	
	res += (alpha - 1) * const;
	return -res; 


def mle():
	global alpha,beta,Gamma,Phi,doc,timer,doc_cnt;
	res = 0;
	timer = 0;
	for d in range(doc_size):
		res += compute_mle(d);
	return res;

def Estep(d, max_iter):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	for n in range(len(doc[d])):
		for i in range(K):
			Phi[d][n][i] = 1 / K;
	for i in range(K):
		Gamma[d][i] = 1/K + alpha;
	last = 0;
	now = compute_mle(d);
	origin = now;
	iter_num = 0;
	while(abs(last - now) > 1e-9 and iter_num < max_iter):

		gamma_sum = sum(Gamma[d]);
		for n in range(len(doc[d])):
			tot = 0;
			for i in range(K):
				Phi[d][n][i] = doc_cnt[d][n] * beta[i][doc[d][n]] * math.exp(sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));
				tot += Phi[d][n][i];
			for i in range(K):
				Phi[d][n][i] = Phi[d][n][i] / tot;
		for i in range(K):
			Gamma[d][i] = 0;
			for n in range(len(doc[d])):
				Gamma[d][i] += doc_cnt[d][n] * Phi[d][n][i];
			Gamma[d][i] += alpha;
		last = now;
		now = compute_mle(d);
		iter_num += 1;

	if (now < origin):
		print('error ' + str(d));

def backtrack(x,dx,df,alpha,beta):
	t = 1;
	while (x + t * dx < 0 or x + t * dx > 1 or compute_alpha_mle(x + t * dx) > compute_alpha_mle(x) + alpha * t * df * dx):
		t = t * beta;
	x = x + t * dx;
	return x;

	
def Mstep(max_iter):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	#update beta
	for i in range(K):
		for v in range(voca_size):
			beta[i][v] = 0;
		for d in range(doc_size):
			for n in range(len(doc[d])):
				beta[i][doc[d][n]] += doc_cnt[d][n] * Phi[d][n][i];
	beta_sum = sum_matrix(beta, 0);
	for i in range(voca_size):
		for k in range(K):
			beta[k][i] = beta[k][i]/beta_sum[i];
	#update alpha
	last = 0;
	iter_num = 0;
	const = 0;
	for d in range(doc_size):
		gamma_sum = sum_vector(Gamma[d]);
		for i in range(K):
			const += (sp.psi(Gamma[d][i]) - sp.psi(gamma_sum));	
	now = -compute_alpha_mle(alpha);
	origin = now;
	while (abs(last - now) > 1e-9 and iter_num < max_iter):
		da = K * (doc_size * (sp.psi(alpha * K) - sp.psi(alpha))) + const;
		dda = K * (doc_size * (K * sp.polygamma(1, alpha * K) - sp.polygamma(1, alpha)));
		dx = -da/dda;
		alpha = backtrack(alpha,dx,da,0.01,0.5);
		last = now;
		now = -compute_alpha_mle(alpha);
		iter_num += 1;
	if (now < origin):
		print('error alpha');

def savemodel(num):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	fout = open('lda_model'+ str(num) + '.txt', 'w');
	fout.write(str(alpha) + '\n');
	for i in range(K):
		for v in range(voca_size):
			fout.write(str(beta[i][v]) + ' ');
		fout.write('\n');



def train(max_iter):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	for i in range(max_iter):
		now = mle();
		print(now);
		print('Estep');
		for d in range(doc_size):
			Estep(d, 20); # the e step of em algorithm
			if (d % 100 == 0):
				print('*');
		now = mle();
		print(now);
		print('Mstep');
		Mstep(20); # the m step of em algorith
		print(alpha);
		savemodel(i);


if __name__ == '__main__':
	fin = readfile('ap.dat');
	doc, voca, doc_cnt = fin.read();
	voca_size = len(voca);
	doc_size = len(doc);
	init();
	train(10);