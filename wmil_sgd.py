#    Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of wmil-sgd.
#
#    wmil-sgd is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
# 
#    wmil-sgd is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with wmil-sgd. If not, see http://www.gnu.org/licenses/

import time
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator
from scipy.sparse import lil_matrix, hstack
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SGDWeights(BaseEstimator):

	def __init__(self, ada=False, alpha=0.2, momentum=0.0, minib=50):
		self.adagrad = ada
		self.alpha = alpha
		self.mom = momentum
		self.minib = minib
		self.gradcheck = False

	def predict(self, B, validate=False):
		pred = []
		if not validate:
			B = self.intercept_sparse(B)
 		for bi in B:
 			pred.append(self.hypo(bi)[0][0])
		return np.array(pred)

 	def predict_weights(self, B, norm=False):
		preds = []
		weights = []
		B = self.intercept_sparse(B)
		for bi in B:
			x = bi.dot(self.O)
			e_x = np.exp(x - np.max(x))
			w = e_x / e_x.sum()
			x_out = bi.T.dot(w)
			p = 0.5*self.W.T.dot(x_out)
			preds.append(p[0][0])
			weights.append(np.array([ww[0] for ww in w]))
		return weights, preds

	def fit(self, X, Y):
		B = self.intercept_sparse(X)
		self.W = np.zeros((B[0].shape[1], 1), dtype=np.float64)
		self.O = np.zeros((B[0].shape[1], 1), dtype=np.float64)
		curb, epoch, maxiter, prev_mae = 0, 0, 500, 99999
		converged = False
		total_sec = []

		print "[+] Training..."
		if self.adagrad:
			gw = 0
			go = 0
			fudge = 1e-6
			epsilon = 0.0
		else:
			vw = self.W.copy()
			vo = self.O.copy()

		while not converged:
			curb = 0
			start = time.time()
			while( curb < len(B) ):
				sum_errw = 0.0
				sum_erro = 0.0
				tob = curb + self.minib
				for i, bi in enumerate(B[curb:tob]):
					sum_errw += self.d_W(bi, Y[curb:tob][i])
					sum_erro += self.d_O(bi, Y[curb:tob][i])
					if self.gradcheck:
						self.grad_check_o(bi, Y[curb:tob][i])
						self.grad_check_w(bi, Y[curb:tob][i])
				sum_errw = sum_errw/(self.minib*1.0)
				sum_erro = sum_erro/(self.minib*1.0)
				if not self.adagrad:
					vw = self.mom * vw - self.alpha * sum_errw
					vo = self.mom * vo - self.alpha * sum_erro
					self.W  += -self.mom * sum_errw + (1 + self.mom) * vw
					self.O  += -self.mom * sum_erro + (1 + self.mom) * vo
	 			else:
					sum_errw += epsilon * self.W
					sum_erro += epsilon * self.O
				 	gw += pow(sum_errw,2)
				 	go += pow(sum_erro,2)
					adjusted_w = sum_errw / (fudge + np.sqrt(gw))
					adjusted_o = sum_erro / (fudge + np.sqrt(go))
					self.W = self.W - self.alpha*adjusted_w
					self.O = self.O - self.alpha*adjusted_o

				curb += self.minib

			pred = self.predict(B, validate=True)
			cur_mae = mean_absolute_error(pred, Y)
			elapsed = (time.time() - start)
			total_sec.append(elapsed)
			print "epoch -> %d / mae: %.6f (%.2f sec)" % (epoch, cur_mae, elapsed)
			if prev_mae - cur_mae < 0.0001 or epoch > maxiter:
				converged = True
			if epoch > 0 and not self.adagrad:
				self.alpha *= 0.998
			prev_mae = cur_mae
			epoch += 1

		tot = sum(total_sec)
		avg = (sum(total_sec)/(1.0*len(total_sec)))
		print "total time: %.2f" % tot
		print "time/epoch: %.2f" % avg
		return 	tot, avg, epoch

	def intercept_sparse(self, X):
		new_x = []
		for i,xi in enumerate(X):
			intercept = np.ones((xi.shape[0],1))
			new_x.append(hstack([intercept,xi]).tocsr())
		return new_x

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		out = e_x / e_x.sum()
		return out

	def mul(self, bi):
		weights = self.softmax(bi.dot(self.O))
		return bi.T.dot(weights)

	def hypo(self, bi):
		x_out = self.mul(bi)
		return 0.5*self.W.T.dot(x_out)

	def d_W(self, bi, yi):
		df_dw = self.mul(bi)
		dl_df = (0.5*self.W.T.dot(df_dw) - yi).view(np.ndarray)[0][0]
		return (dl_df * df_dw)

	def d_O(self, bi, yi):
		pi = self.softmax(bi.dot(self.O))
		x_out = bi.T.dot(pi)
		dl_df = (0.5*self.W.T.dot(x_out) - yi)
		df_dg = bi.dot(self.W)
 		dg_do = (np.identity(bi.shape[0])-pi)*pi
		return bi.T.dot((dl_df*df_dg).T.dot(dg_do).T)

	def L(self, bi, yi, W, O):
		self.W, self.O = W, O
		return pow((self.hypo(bi) - yi),2)

	def grad_check_w(self, bi, yi):
		w_e = self.W
		epsilon = 0.00001
		for i in range(len(w_e)):
			cur_p = w_e.copy()
			cur_p[i] = cur_p[i] + epsilon
			cur_m = w_e.copy()
			cur_m[i] = cur_m[i] - epsilon
			actual = (self.L(bi,yi, cur_p, self.O) - self.L(bi, yi, cur_m, self.O))/(2.0*epsilon)
			approx = self.d_W(bi, yi)[i]
			if actual != 0:
				nom = abs(approx - actual)[0][0]
				denom = np.max([approx[0], actual[0][0]])
				if nom/denom < 0.0001:
				 	print "[+] approx: %.8f / actual: %.8f " % (approx, actual)
				else:
					print "[-] approx: %.8f / actual: %.8f " % (approx, actual)

	def grad_check_o(self, bi, yi):
		o_e = self.O
		epsilon = 0.00001
		print
		for i in range(len(o_e)):
			cur_p = o_e.copy()
			cur_p[i] = cur_p[i] + epsilon
			cur_m = o_e.copy()
			cur_m[i] = cur_m[i] - epsilon
			actual = (self.L(bi,yi, self.W, cur_p) - self.L(bi, yi, self.W, cur_m))/(2.0*epsilon)
			approx = self.d_O(bi, yi)[i]
			if actual != 0 or approx !=0:
				nom = abs(approx - actual)[0][0]
				denom = np.max([approx[0], actual[0][0]])
				if abs(approx - actual)[0][0] < 0.0001:
				 	print "[+] -> (%d) approx: %.10f / actual: %.10f " % (i, approx, actual)
				else:
					print "[-] -> (%d) approx: %.10f / actual: %.10f " % (i, approx, actual)
