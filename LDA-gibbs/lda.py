from readfile import readfile;
import scipy.special as sp;
import math;
import time;
import random;

alpha = 0;
beta = 1;
K = 10;
Gamma = {};
Phi = {};
doc = {};
doc_cnt = [];
doc_size = 0;
voca_size = 0;
timer = 0;
Z = [];
word_topic = [];
doc_topic = [];
topic_cnt = [];
Beta = [];
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
	global alpha,beta,Gamma,Phi,doc,doc_cnt,Z,word_topic,doc_topic,topic_cnt,Beta;

	#random start
	beta = 1;
	Beta = [];
	for i in range(K):
		Beta.append([]);
		for j in range(voca_size):
			Beta[i].append(0);

	Z = [];
	for d in range(doc_size):
		Z.append([]);
		for i in range(len(doc[d])):
			Z[d].append(0);

	Gamma = [];

	for k in range(K):
		topic_cnt.append(0);
		word_topic.append([]);
		doc_topic.append([]);
		for v in range(voca_size):
			word_topic[k].append(0);
		for d in range(doc_size):
			doc_topic[k].append(0);

	for d in range(doc_size):
		Gamma.append([]);
		for k in range(K):
			Gamma[d].append(0);

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


	for n in range(length):
		for i in range(K):
			res += doc_cnt[d][n] * Phi[d][n][i] * psi[i];
			res += doc_cnt[d][n] * Phi[d][n][i] * math.log(beta[i][doc[d][n]]/Phi[d][n][i]);

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
	for d in range(doc_size):
		res += compute_mle(d);
	return res;

def sample_topic(d,i):
	global alpha,beta,Gamma,Phi,doc,doc_cnt,Z,word_topic,doc_topic,topic_cnt;
	topic = Z[d][i];
	word = doc[d][i];
	cnt = doc_cnt[d][i];
	topic_cnt[topic] -= cnt;
	doc_topic[topic][d] -= cnt;
	word_topic[topic][word] -= cnt;

	P = [];
	tot = 0;
	for i in range(K):
		p = ((word_topic[topic][word] + beta)/(topic_cnt[i] + voca_size * beta)) * ((doc_topic[topic][d] + alpha)/ (len(doc[d]) - 1 + K * alpha));
		P.append(p);
		tot += p;
	for i in range(K):
		P[i] /= tot;

	r = random.random();
	for i in range(K):
		r -= P[i];
		if (r < 0):
			topic = i;
			break;

	topic_cnt[topic] += cnt;
	doc_topic[topic][d] += cnt;
	word_topic[topic][word] += cnt;
	return topic;


def Estep(max_iter):
	global alpha,Beta,Gamma,Phi,doc,doc_cnt,Z,word_topic,doc_topic,topic_cnt;

	for k in range(K):
		topic_cnt[k] = 0;
		for d in range(doc_size):
			doc_topic[k][d] = 0;
		for v in range(voca_size):
			word_topic[k][v] = 0;

	for d in range(doc_size):
		for i in range(len(doc[d])):
			Z[d][i] = random.randint(0,K-1);
			topic_cnt[Z[d][i]] += doc_cnt[d][i];
			doc_topic[Z[d][i]][d] += doc_cnt[d][i];
			word_topic[Z[d][i]][doc[d][i]] += doc_cnt[d][i];

	for iter_num in range(max_iter):
		#Gibbs sampling
		if (iter_num % 10 == 0):
			print('*');
		for d in range(doc_size):
			for i in range(len(doc[d])):
				Z[d][i] = sample_topic(d,i);

	#update Beta, Gamma
	for k in range(K):
		for v in range(voca_size):
			Beta[k][v] = (word_topic[k][v] + beta)/(topic_cnt[k] + voca_size * beta);
	
	for d in range(doc_size):
		for k in range(K):
			Gamma[d][k] = (doc_topic[k][d] + alpha)/(len(doc[d]) + K * alpha);
			



def backtrack(x,dx,df,alpha,beta):
	t = 1;
	while (x + t * dx < 0 or x + t * dx > 1 or compute_alpha_mle(x + t * dx) > compute_alpha_mle(x) + alpha * t * df * dx):
		t = t * beta;
	x = x + t * dx;
	return x;

	
def Mstep(max_iter):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
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
	global alpha,Beta,Gamma,Phi,doc,doc_cnt;
	fout = open('lda_model'+ str(num) + '.txt', 'w');
	fout.write(str(alpha) + '\n');
	for i in range(K):
		for v in range(voca_size):
			fout.write(str(Beta[i][v]) + ' ');
		fout.write('\n');



def train(max_iter):
	global alpha,beta,Gamma,Phi,doc,doc_cnt;
	for i in range(max_iter):
		print('Estep');
		Estep(100);
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