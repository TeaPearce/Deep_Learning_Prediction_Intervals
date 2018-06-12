# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import math

"""
this is where the magic happens:
- defines NN structure
- defines NN loss fn
- training loop
- visualise training
also has a lot of functions toward the bottom that provide the engine for
the particle swarm optimisation training method (run from pso.py)
ignore anything to do with censoring - this was due to an experiment that didn't work
I tried playing around with running things on cpu vs gpu to get speed up
it didn't make much difference for my experiments so stayed with cpu
but if you have tensorflow set up with a gpu, change cpu->gpu in statements: with tf.device("/cpu:0")
"""

class TfNetwork:
	def __init__(self, x_size, y_size, h_size, type_in="pred_intervals", 
		alpha=0.1, loss_type='qd_soft', censor_R=False, 
		soften=100., lambda_in=10., sigma_in=0.5, activation='relu', bias_rand=False, out_biases=[2.,-2.],
		**kwargs):
		"""
		sets up network with
		x_size is integer no. of input variables
		y_size is integer no. of output nodes
		h_size is list of integers of hidden layers, must be at least 1
		type_in just use: pred_intervals (originally was going to allow classification etc)
		alpha is % of samples ant to exclude
		loss type defines which loss drives training
		censor_R chooses whether to set up loss for censoring
		kwargs is kind of placeholder in case get more complicated
		 - add dropout, regularisation etc
		"""

		self.x_size = x_size
		self.y_size = y_size
		self.h_size = h_size
		self.type_in = type_in # run some validation for these
		self.soften = soften
		self.lambda_in = lambda_in

		# set up input and output
		X = tf.placeholder(tf.float32, [None,x_size])
		y_true = tf.placeholder(tf.float32, [None,1]) # force to one for PI
		
		# set up censoring input
		cens_R = tf.placeholder(tf.float32, [None,1]) 
		# will be a dumby var if not using censoring
			# True = censored = 1.
			# False = observed = 0.

		# set up parameters
		W = [] # list of variables
		b = []
		layer_in = [] # before activation
		layer = [] # post-activation
		sigma = sigma_in # stddev of initialising norm dist

		# first layer
		W.append(tf.Variable(tf.random_normal([x_size, h_size[0]], stddev=sigma)))
		if bias_rand: # add some noise - useful for pso
			b.append(tf.Variable(tf.random_normal([h_size[0]], mean=0.1, stddev=sigma/5.)))
		else:
			b.append(tf.Variable(np.zeros(h_size[0])+0.1, dtype=tf.float32))

		# add hidden layers
		for i in range(1,len(h_size)):
			W.append(tf.Variable(tf.random_normal([h_size[i-1], h_size[i]], stddev=sigma)))
			if bias_rand: # add some noise - useful for pso
				b.append(tf.Variable(tf.random_normal([h_size[i]], mean=0.1, stddev=sigma/5.)))
			else:
				b.append(tf.Variable(np.zeros(h_size[i])+0.1, dtype=tf.float32))

		# add final layer
		W.append(tf.Variable(tf.random_normal([h_size[-1], y_size], stddev=sigma)))
		if loss_type == 'gauss_like' or loss_type == 'mse':
			b.append(tf.Variable([0.0,1.0]))
		else:
			b.append(tf.Variable(out_biases)) # for LUBE

		# if relu it's useful to add a little bit
		# for i in range(0,len(b)):
		# 	if bias_rand: # add some noise - useful for pso
		# 		b[i] = b[i]+np.random.normal(loc=0.1,scale=sigma/5.)
		# 	else:
		# 		b[i] = b[i]+0.1

		# define model - first layer
		with tf.device("/cpu:0"): # change all cpu to gpu if want, but 
		# didn't make much faster in my tests
			layer_in.append(tf.matmul(X, W[0]) + b[0])
			if activation == 'relu':
				layer.append(tf.nn.relu(layer_in[-1]))
			elif activation == 'tanh':
				layer.append(tf.nn.tanh(layer_in[-1]))

		# hidden layers
		for i in range(1,len(h_size)):
			with tf.device("/cpu:0"):
				layer_in.append(tf.matmul(layer[i-1], W[i]) + b[i])
				if activation == 'relu':
					layer.append(tf.nn.relu(layer_in[-1]))
				elif activation == 'tanh':
					layer.append(tf.nn.tanh(layer_in[-1]))

		# create metric list (so can have as many as want)
		metric = []
		metric_name = []

		# final layer
		if self.type_in == "pred_intervals_simple":
			# reset the network - only want one weight
			W = [] # list of variables
			b = []
			layer_in = [] # before activation
			layer = [] # post-activation

			W.append(tf.placeholder(tf.float32, [1])) # for PI
			b.append(tf.Variable(tf.zeros([1])))

			layer_in.append(tf.multiply(X, W[0]))
			y_pred = layer_in[-1] # since it's linear no need for fn

			y_U = y_pred[:,0]
			y_T = y_true[:,0]
			alpha_ = tf.constant(alpha)

			# unsym abs loss
			c_1 = (1-alpha_)/alpha_-0.0001
			loss = tf.reduce_mean(tf.maximum(tf.subtract(y_U,y_T),c_1*tf.subtract(y_T,y_U)))

			# one sided LUBE
			N_ = 10. # sample size
			c_ = tf.constant(1.) # could run a pred to get this first
			lambda_ = tf.constant(1.) 
			gamma_U = tf.maximum(0.,tf.sigmoid((y_U - y_T)*100.))
			# gamma_U = tf.maximum(0.,tf.sign(y_U - y_T))
			gamma_ = gamma_U
			ones_ = tf.ones_like(gamma_)
			# loss = tf.reduce_mean(tf.abs(y_U - y_T)*gamma_U)+lambda_*c_*tf.maximum(0., tf.reduce_sum((ones_-gamma_U)/(alpha_*N_) - (ones_/N_)))

			self.loss = loss

			metric.append(loss)
			metric_name.append("dummy")


		elif self.type_in == "pred_intervals":

			# finish defining network
			with tf.device("/cpu:0"):
				layer_in.append(tf.matmul(layer[-1], W[-1]) + b[-1])
			y_pred = layer_in[-1] # since it's linear no need for fn

			# get components
			y_U = y_pred[:,0]
			y_L = y_pred[:,1]
			y_T = y_true[:,0]

			# set inputs and constants
			N_ = tf.cast(tf.size(y_T),tf.float32) # sample size
			alpha_ = tf.constant(alpha)
			#lambda_ = tf.placeholder(tf.float32, [1]) # this could be a constant or anneal it
			lambda_ = tf.constant(lambda_in) 

			with tf.device("/cpu:0"):
				
				# --- DEFINE THE LOSS FUNCTION ---

				# in case want to do point predictions
				y_pred_mean = tf.reduce_mean(y_pred,axis=1)
				MPIW = tf.reduce_mean(tf.subtract(y_U,y_L))

				# soft uses sigmoid
				gamma_U = tf.sigmoid((y_U - y_T)*soften)
				gamma_L = tf.sigmoid((y_T - y_L)*soften)
				gamma_ = tf.multiply(gamma_U,gamma_L)
				ones_ = tf.ones_like(gamma_)

				# hard uses sign step fn
				gamma_U_hard = tf.maximum(0.,tf.sign(y_U - y_T))
				gamma_L_hard = tf.maximum(0.,tf.sign(y_T - y_L))
				gamma_hard = tf.multiply(gamma_U_hard,gamma_L_hard)

				# lube - lower upper bound estimation
				qd_lhs_hard = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L)*gamma_hard), tf.reduce_sum(gamma_hard)+0.001)
				qd_lhs_soft = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L)*gamma_), tf.reduce_sum(gamma_)+0.001) # add small noise in case 0
				PICP_soft = tf.reduce_mean(gamma_)
				PICP_hard = tf.reduce_mean(gamma_hard)
				qd_rhs_soft = lambda_*tf.sqrt(N_)* tf.square(tf.maximum(0., (1. - alpha_) - PICP_soft))
				qd_rhs_hard = lambda_*tf.sqrt(N_)* tf.square(tf.maximum(0., (1. - alpha_) - PICP_hard))
				# old method
				qd_loss_soft = qd_lhs_hard + qd_rhs_soft # full LUBE w sigmoid for PICP
				qd_loss_hard = qd_lhs_hard + qd_rhs_hard # full LUBE w step fn for PICP

				umae_loss = 0 # ignore this

				# gaussian log likelihood
				# already defined output nodes 
				# y_U = mean, y_L = variance
				y_mean = y_U

				# from deep ensemble paper
				y_var_limited = tf.minimum(y_L, 10.) # seem to need to limit otherwise causes nans occasionally
				y_var = tf.maximum(tf.log(1.+tf.exp(y_var_limited)), 10e-6)

				# to track nans
				self.y_mean = y_mean
				self.y_var = y_var

				gauss_loss = tf.log(y_var)/2. + tf.divide(tf.square(y_T-y_mean), 2.*y_var) # this is -ve already
				gauss_loss = tf.reduce_mean(gauss_loss) 
				# use mean so has some kind of comparability across datasets
				# but actually need to rescale and add constant if want to get actual results


			# set main loss type
			if loss_type == 'qd_soft':
				loss = qd_loss_soft			
			elif loss_type == 'qd_hard':
				loss = qd_loss_hard
			elif loss_type =='umae_R_cens':
				loss = umae_loss_cens_R
			elif loss_type == 'gauss_like':
				loss = gauss_loss
			elif loss_type == 'picp': # for loss visualisation
				loss = PICP_hard
			elif loss_type =='mse':
				loss = tf.reduce_mean(tf.squared_difference(y_U, y_T))

			# add metrics
			with tf.device("/cpu:0"):
				u_capt = tf.reduce_mean(gamma_U_hard) # apparently is quicker if define these
				l_capt = tf.reduce_mean(gamma_L_hard) # here rather than in train loop
				all_capt = tf.reduce_mean(gamma_hard)
			# metric.append(u_capt)
			# metric_name.append('U capt.')			
			# metric.append(l_capt)
			# metric_name.append('L capt.')			
			metric.append(all_capt)
			metric_name.append('PICP')			
			metric.append(MPIW)
			metric_name.append('MPIW')
			# metric.append(tf.reduce_mean(tf.pow(y_T - y_pred_mean,2)))
			# metric_name.append("MSE mid")

			# store
			self.y_U = y_U
			self.y_L = y_L
			self.y_T = y_T
			self.y_pred_mean = y_pred_mean
			self.gamma_U = gamma_U
			self.gamma_L = gamma_L
			self.gamma_ = gamma_
			self.qd_lhs_soft = qd_lhs_soft		
			self.qd_lhs_hard = qd_lhs_hard
			self.lube_rhs = qd_rhs_hard
			self.lube_loss = qd_loss_soft
			self.umae_loss = umae_loss
			self.loss = loss

		# so can access gradients later
		# grad_wrt_v = []
		# for i in range(0,len(W)):
		# 	grad_wrt_v.append(tf.gradients(loss, W[i])[0])
		# for i in range(0,len(b)):
		# 	grad_wrt_v.append(tf.gradients(loss, b[i])[0])
		# grad_wrt_v.append(tf.gradients(loss, y_pred)[0])
		# grad_wrt_v.append(tf.gradients(loss, layer_in[-1])[0])

		# store
		self.X = X
		self.y_pred = y_pred
		self.y_true = y_true
		self.cens_R = cens_R
		self.loss = loss
		self.metric = metric
		self.metric_name = metric_name
		self.W = W
		self.b = b
		self.layer = layer
		self.layer_in = layer_in
		# self.grad_wrt_v = grad_wrt_v
		self.loss_type = loss_type

		# TODO...
		#if dropout etc in kwargs.keys():


	def train(self, sess, X_train, y_train, X_val, y_val, n_epoch, l_rate=0.01,
		resume_train=False, print_params=True, decay_rate=0.95, 
		censor_R_ind=None, is_early_stop=False, is_use_val=False, optim='SGD',
		is_batch=False, n_batch=100, is_run_test=False, is_print_info=True, **kwargs):
		"""
		do training schedule
		"""
		# if no censoring make a dumby variable
		if censor_R_ind is None:
			censor_R_ind = np.zeros_like(X_train[:,0])[:,np.newaxis]

		# set up session
		if False: # trying to avoid need for new sessions
			sess = tf.InteractiveSession()

		global_step = tf.Variable(0, trainable=False) # keep track of which epoch on
		decayed_l_rate = tf.train.exponential_decay(l_rate, global_step,
			decay_steps=50, decay_rate=decay_rate, staircase=False)
		# eqn: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
		with tf.device("/cpu:0"):
			if optim == 'SGD':
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=decayed_l_rate)
			elif optim =='adam':
				optimizer = tf.train.AdamOptimizer(learning_rate=decayed_l_rate)
			else:
				raise Exception('ERROR unusual optimizer specified')

			train_step = optimizer.minimize(self.loss, global_step=global_step)

		saver = tf.train.Saver()
		if resume_train:
			saver.restore(sess, "/tmp/model.ckpt") # load variables
			loss_log = self.loss_log.tolist() # convert log back to list type
		else:
			sess.run(tf.global_variables_initializer()) # init variables
			loss_log = []
		
		# print params
		if print_params:
			print('\n--- params before training ---')
			self.print_params(sess, in_sess = True)
		
		# train
		loss_train_prev=1000.
		seq_smaller=0
		for epoch in range(0,n_epoch):
			
			if is_batch:
				# shuffle order
				perm = np.random.permutation(X_train.shape[0])
				X_train_shuff = X_train[perm]
				y_train_shuff = y_train[perm]
				censor_R_ind_shuff = censor_R_ind[perm]

				loss_train=0
				# for each batch
				n_batches = int(round(X_train.shape[0]/n_batch))
				for b in range(0,n_batches):

					# if last batch use all data
					if b == int(round(X_train.shape[0]/n_batch)):
						X_train_b = X_train_shuff[b*n_batch:]
						y_train_b = y_train_shuff[b*n_batch:]
						c_train_b = censor_R_ind_shuff[b*n_batch:]
					else:
						X_train_b = X_train_shuff[b*n_batch:(b+1)*n_batch]
						y_train_b = y_train_shuff[b*n_batch:(b+1)*n_batch]
						c_train_b = censor_R_ind_shuff[b*n_batch:(b+1)*n_batch]
					_, loss_train_b = sess.run([train_step, self.loss], feed_dict={self.X: X_train_b, self.y_true: y_train_b, self.cens_R: c_train_b})
					loss_train += loss_train_b/n_batches
			else:
				# whole dataset
				_, loss_train = sess.run([train_step, self.loss], feed_dict={self.X: X_train, self.y_true: y_train, self.cens_R: censor_R_ind})
			

			# whether to print info or not
			if is_run_test:
				is_print=False
				# give details for every x, and the last run
				if epoch % int(n_epoch/10) == 0 or epoch == n_epoch-1 : 
					is_print=True
			else: 
				# for development
				is_print = True

			# print info
			if is_print:
				# loss_train = sess.run(self.loss, 
				# 	feed_dict={self.X: X_train, self.y_true: y_train, self.cens_R: censor_R_ind})
				
				if is_use_val:
					loss_val, l_rate_epoch = sess.run([self.loss, decayed_l_rate], 
						feed_dict={self.X: X_val, self.y_true: y_val})
				else:
					loss_val = loss_train # quicker for training
					l_rate_epoch = sess.run(decayed_l_rate, 
						feed_dict={self.X: X_val, self.y_true: y_val})
				
				if is_print_info:
					print('\nep:',epoch, ' \ttrn loss',round(loss_train,4),'  \tval loss', round(loss_val,4), end='\t')

				# the metrics don't really make sense for gauss likelihood
				if self.loss_type != 'gauss_like':
					if True:
					# if epoch % int(n_epoch/50) == 0:
						ops_to_run=[]
						for i in range(0,len(self.metric)):
							ops_to_run.append(self.metric[i])
						_1, _2 = sess.run(ops_to_run, feed_dict={self.X: X_val, self.y_true: y_val, self.cens_R: censor_R_ind})

						if is_print_info:
							print(self.metric_name[0], round(_1,4), '\t',
								self.metric_name[1], round(_2,4), 
								end='\t')
				if is_print_info:
					print('l_rate', round(l_rate_epoch,5), end='\t')
				# print('l_rate', round(sess.run(decayed_l_rate),5), end='\t')
				loss_log.append((epoch,loss_train,loss_val))



				# searching for cause of nan
				# self.print_params(sess, in_sess = True)
				# if math.isnan(loss_val):
				if False:
					 nan_mean = sess.run(self.y_mean, 
						feed_dict={self.X: X_val, self.y_true: y_val})
					 nan_var = sess.run(self.y_var, 
						feed_dict={self.X: X_val, self.y_true: y_val})
					 nan_y_L = sess.run(self.y_L, 
						feed_dict={self.X: X_val, self.y_true: y_val})

					 # print('nan_mean', nan_mean)
					 # print('nan_var', nan_var)
					 # print('nan_y_L', nan_y_L)

					 # self.print_params(sess, in_sess = True)


				# check for stopping criteria - just fast simple method
				# if reduced for x steps without noise should mean is stable
				if is_early_stop:
					if loss_train < loss_train_prev:
						seq_smaller += 1
					else:
						seq_smaller=0 # reset
					
					# if has reduced for 100 epochs
					if seq_smaller > 20 and epoch > n_epoch/2:
						epoch = n_epoch - 1
						break
					loss_train_prev = loss_train


		# print params
		if print_params:
			print('\n--- params after training ---')
			self.print_params(sess, in_sess = True)

		# tidy up
		if False: # trying to avoid need for new sessions
			save_path = saver.save(sess, "/tmp/model.ckpt")
			sess.close()

		self.loss_log = np.array(loss_log) # convert list to array
		self.last_loss_trn = self.loss_log[-1,1]

		return


	def print_params(self, sess, in_sess=False):
		"""
		short method to print NN parameters
		in_sess tells us if we need to enter a new session or are already in one
		"""
		if not in_sess:
			sess = tf.InteractiveSession()
			saver = tf.train.Saver()
			saver.restore(sess, "/tmp/model.ckpt")

		update_op=[]
		for i in range(0,len(self.W)):
			update_op.append(self.W[i])
			print("W[ %i ]:\n" % i)
		for i in range(0,len(self.b)):
			update_op.append(self.b[i])
			print("b[ %i ]:\n" % i)

		params_all = sess.run(update_op)
		print(params_all)

		if not in_sess:
			sess.close()

		return


	def return_params_list(self, sess, in_sess=False):
		"""
		return value of params - for pso training
		we needed this one to return as list of arrays
		avoid messing previous things up by using new fn
		"""

		# slow version
		# params_all = []
		# for i in range(0,len(self.W)):
		# 	params_all.append(self.W[i].eval())
		# for i in range(0,len(self.b)):
		# 	params_all.append(self.b[i].eval())

		# print('params_all slow', params_all)

		# quicker version
		update_op=[]
		for i in range(0,len(self.W)):
			update_op.append(self.W[i])
		for i in range(0,len(self.b)):
			update_op.append(self.b[i])

		params_all = sess.run(update_op)

		return params_all


	def set_params(self, params_new, sess):
		"""
		update NN params from np vector
		for pso scheme
		"""
		update_op=[]
		for i in range(0,len(self.W)):
			update_op.append(self.W[i].assign(params_new[i]))
		for i in range(0,len(self.b)):
			update_op.append(self.b[i].assign(params_new[i+len(self.W)]))
		
		# running these in one update seems to be much quicker			
		sess.run(update_op)

		return


	def pso_init(self, X, y, sess, v_max, name):
		"""
		initialise each particle
		"""

		# find current info
		p_loss = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y})
		p_params = self.return_params_list(sess, in_sess=True)

		# initalise velocity for each param of each particle
		# rand_vel = np.random.uniform(low=-v_max,high=v_max,size=p_params) # this was wrong shape
		rand_vel = []
		for i in range(0,len(p_params)):
			rand_vel.append(np.random.uniform(low=-v_max,high=v_max,size=p_params[i].shape))
		
		# debug
		# print(p_params)
		# print('rand_vel', rand_vel)

		# store
		self.name = name # useful for keeping track particles
		self.curr_params = p_params
		self.best_params = p_params
		self.curr_loss = p_loss
		self.best_loss = p_loss
		self.velocity = rand_vel

		return


	def pso_update(self, X, y, sess, v_max, v_prev, v_local, v_global, 
		best_global_params):
		"""
		run update for training via pso
		"""
		v_local = 1.0
		v_global = 1.2

		# pso algorithm
		rand_locl = []
		rand_glob = []
		velocity_new = []
		params_new = []
		for i in range(0,len(self.curr_params)):
			rand_locl.append(np.random.uniform(low=0.,high=1.,size=self.curr_params[i].shape))
			rand_glob.append(np.random.uniform(low=0.,high=1.,size=self.curr_params[i].shape))
			velocity_new.append(v_prev*self.velocity[i] + \
				v_local*rand_locl[i]*(self.best_params[i]-self.curr_params[i]) + \
				v_global*rand_glob[i]*(best_global_params[i]-self.curr_params[i]))
			# params_new.append(self.curr_params[i] + velocity_new[i])

			# add some noise - found uniform works best
			params_new.append(self.curr_params[i] + velocity_new[i] + np.random.uniform(low=-0.01,high=0.01)) # used to use 0.2
			# params_new.append(self.curr_params[i] + velocity_new[i] + np.random.normal(loc=0.0,scale=0.01))

		# debug
		# print('rand_locl',rand_locl)
		# print('rand_glob',rand_glob)
		# print('velocity_new',velocity_new)
		# print('params_new',params_new)

		# velocity_new = v_prev*self.velocity + \
		# 	v_local*rand_locl*(self.best_params-self.curr_params) + \
		# 	v_global*rand_glob*(best_global_params-self.curr_params)

		# params_new = self.curr_params + velocity_new

		# update NN params
		# self.set_params(params_new, sess)

		# could break loop here, return to parent loop, run sess for all p's
		# then continue

		# update particle info
		# p_loss = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y})


		# debug
		# print('type:',type(params_new))
		# print('type params[0]:',type(params_new[0]))
		# print('params[0]:',params_new[0])
		# print('type params[0][0]:',type(params_new[0][0]))
		# print('params[0][0]:',params_new[0][0])
		# print('params_new[:len(self.W)]', params_new[:len(self.W)])
		# print('self.W', self.W)
		# print('params_new[len(self.W):]', params_new[len(self.W):])
		# print('self.b', self.b)

		# p_loss = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y,
		# 	self.W: params_new[:len(self.W)], 
		# 	self.b: params_new[len(self.W):] })
		p_loss = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y,
			self.W[0]: params_new[0],
			self.W[1]: params_new[1],
			# self.W[2]: params_new[2],
			self.b[0]: params_new[2],
			self.b[1]: params_new[3],})
			# self.b[2]: params_new[5] })
		
		# debugging
		# print('\nparticle_name:', self.name)
		# print('self.curr_params:', self.curr_params)
		# print('params_new:', params_new)
		# print('loss_new:', p_loss)
		# print('particle:', self.name, '\tloss_new:', p_loss)

		self.curr_params = params_new
		self.curr_loss = p_loss
		self.velocity = velocity_new
		if p_loss < self.best_loss:
			self.best_params = params_new
			self.best_loss = p_loss

		return


	def predict(self, sess, X, y, censor_R_ind=None, in_sess=False):
		"""
		run prediction
		not forced to supply a y value incase doing genuine prediciton
		in_sess tells us if we need to enter a new session or are already in one
		"""
		# if no censoring make a dumby variable
		if censor_R_ind is None:
			censor_R_ind = np.zeros_like(X[:,0][:,np.newaxis])

		if not in_sess:
			sess = tf.InteractiveSession()
			saver = tf.train.Saver()
			saver.restore(sess, "/tmp/model.ckpt")

		y_pred_out = sess.run(self.y_pred, feed_dict={self.X: X, self.y_true: y, self.cens_R: censor_R_ind})
		y_metric_out = sess.run(self.metric, feed_dict={self.X: X, self.y_true: y, self.cens_R: censor_R_ind})
		y_loss = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y, self.cens_R: censor_R_ind})
		y_U_cap = y_pred_out[:,0] > y.reshape(-1)
		y_U_prop = np.sum(y_U_cap)/y_U_cap.shape[0]	
		y_L_cap = y_pred_out[:,1] < y.reshape(-1)
		y_L_prop = np.sum(y_L_cap)/y_L_cap.shape[0]
		y_all_cap = y_U_cap*y_L_cap
		y_all_prop = np.sum(y_all_cap)/y_L_cap.shape[0]

		if not in_sess:
			sess.close()

		return y_loss, y_pred_out, y_metric_out, y_U_cap, y_U_prop, y_L_cap, y_L_prop, y_all_cap, y_all_prop, 


	def loss_surface_simple(self, sess, X, y, in_sess=False):
		"""
		loss surface may be visualised
		for simple example, vary one weight and record results
		assumes input network is 1 input node, 1 hidden layer, 2 output nodes
		all weights and biases except one are held at default values
		in_sess tells us if we need to enter a new session or are already in one
		"""
		if not in_sess:
			sess = tf.InteractiveSession()
			saver = tf.train.Saver()
			saver.restore(sess, "/tmp/model.ckpt")

		results_save = []
		for W_in in np.linspace(0.5,1.2,1000):
			y_pred_out = sess.run(self.y_pred, feed_dict={self.X: X, self.y_true: y, 
				self.W[0]: [[W_in]], self.W[1]: [[1.,0.]], 
				self.b[0]: [0.], self.b[1]: [0.,0.15]})
			y_metric_out = sess.run(self.loss, feed_dict={self.X: X, self.y_true: y,
				self.W[0]: [[W_in]], self.W[1]: [[1.,0.]], 
				self.b[0]: [0.], self.b[1]: [0.,0.15]})
			y_U_cap = y_pred_out[:,0] > y.reshape(-1)
			y_U_prop = np.sum(y_U_cap)/y_U_cap.shape[0]
			# print(W_in, y_pred_out[0], y_U_prop, y_metric_out)
			results_save.append([W_in,y_pred_out[0][0],y_U_prop, y_metric_out])

		if not in_sess:
			sess.close()

		return np.array(results_save)


	def vis_train(self, save_graphs=False, is_use_val=False):
		"""
		view graph of training history
		"""
		fig, ax = plt.subplots(1)
		ax.plot(self.loss_log[:,0],self.loss_log[:,1], linewidth=0.5, color='k', label='train')
		if is_use_val:
			ax.plot(self.loss_log[:,0],self.loss_log[:,2], linewidth=0.5, color='r', alpha=0.6, label='test')
			ax.legend(loc='upper right')
		ax.set_xlabel("epochs")
		ax.set_ylabel("loss")
		#ax.plot(NN.loss_log[:,0],NN.loss_log[:,2]) # val
		title = 'loss=' + str(round(self.last_loss_trn,4)) \
				+ ', loss_type=' + self.loss_type + ',' \
				+ '\nh_size=' + str(self.h_size) \
				+ ', soften=' + str(self.soften) \
				+ ', lambda_in=' + str(self.lambda_in)
		ax.set_title(title)
		if save_graphs:
			fig.savefig('02_outputs/fig_grad_train.png', bbox_inches='tight')
		
		fig.show()

		return


class swarm:
	def __init__(self, X, y, X_val, y_val,
		x_size, y_size, h_size, type_in, sess,
		alpha=0.1, loss_type='lube', censor_R=False, 
		soften=100., lambda_in=10., sigma_in=0.5, 
		particle_n=40, v_prev=0.721,v_global=1.193, v_local=1.193, v_max=2.0,
		n_epoch=100, activation='relu', **kwargs):
		"""
		run particle swarm optimisation (pso) training scheme
		we'll use tensorflow stuff in order to mimic as close
		as possible the gradient trained versions
		some inputs are those required to initialise NN
		others are for the PSO scheme
		it will run off of the default loss of NN
		"""

		# initialise
		particles = []
		name = 0
		for i in range(0,particle_n):
			print('creating particle:',name)
			particles.append(TfNetwork(x_size, y_size, h_size, type_in, 
				alpha, loss_type, censor_R, 
				soften, lambda_in, sigma_in, bias_rand=True, activation=activation))
			name += 1

		sess.run(tf.global_variables_initializer())

		# initialisation loop
		best_global_loss = 1000000.
		best_global_params = np.array(1) # dummy array to start
		best_global_name = 0
		name = 0
		for p in particles:

			# sets best position, best params, velocity
			p.pso_init(X, y, sess, v_max, name)
			print('initalising particle:', name, '\tinitial loss:', p.best_loss)
			name += 1

			# initialise global best
			if p.best_loss < best_global_loss:
				best_global_loss = p.best_loss
				best_global_params = p.best_params
				best_global_name = p.name
				best_NN = p

		print('\ncurr best_global_name:',best_global_name)
		print('curr best_global_loss:',best_global_loss)	

		# main loop
		loss_log = []
		for epoch in range(0,n_epoch):
			string_in = '\n-- curr epoch: ' + str(epoch)
			print_time(string_in)
			for p in particles:
				# sets new values
				p.pso_update(X, y, sess, v_max, v_prev, 
					v_local, v_global, best_global_params)

				# print info
				print('\nep ',epoch, ' part ', p.name, '\ttrn loss ',round(p.curr_loss,5), end='\t')
				for i in range(0,len(p.metric)):
					print(p.metric_name[i], 
						sess.run(p.metric[i], feed_dict={p.X: X_val, p.y_true: y_val,
						p.W[0]: p.curr_params[0],
						p.W[1]: p.curr_params[1],
						# p.W[2]: p.curr_params[2],
						p.b[0]: p.curr_params[2],
						p.b[1]: p.curr_params[3]}),
						# p.b[2]: p.curr_params[5] }),
						end='\t')

				# update global best
				if p.best_loss < best_global_loss:
					best_global_loss = p.best_loss
					best_global_params = p.best_params
					best_global_name = p.name
					best_NN = p

				loss_log.append((epoch, p.name, p.curr_loss, best_global_loss))

			print('\ncurr best_global_name:',best_global_name)
			print('curr best_global_loss:',best_global_loss)

		print('best_global_params:',best_global_params)

		best_NN.set_params(best_global_params, sess)

		self.particle_n = particle_n
		self.loss_log = np.array(loss_log)
		self.best_NN = best_NN # return best NN
		self.loss_type = loss_type
		self.alpha = alpha
		self.v_prev = v_prev
		self.v_global = v_global
		self.v_local = v_local
		self.v_max = v_max
		self.best_global_loss = best_global_loss


	def vis_train(self):
			"""
			view graph of training history
			"""
			fig, ax = plt.subplots(1)
			for i in range(0,self.particle_n):
				temp = self.loss_log[self.loss_log[:,1] == i]
				# ax.plot(temp[:,0],temp[:,2], alpha=0.1, color='k') # train
				ax.scatter(temp[:,0],temp[:,2], alpha=0.2, marker='.', s=1.0, color='k') # train
			ax.plot(temp[:,0],temp[:,3], linewidth=1.0, linestyle='--', color='r', ) # train

			ax.set_xlabel("epochs")
			ax.set_ylabel("loss")
			ax.set_yscale('log')
			title = 'loss=' + str(round(self.best_global_loss,4)) \
				+ ', loss_type=' + self.loss_type \
				+ ', v_prev=' + str(self.v_prev) \
				+ ', \nv_global=' + str(self.v_global) \
				+ ', v_max=' + str(self.v_max) \
				+ ', particle_n=' + str(self.particle_n)
			ax.set_title(title)
			#ax.plot(NN.loss_log[:,0],NN.loss_log[:,2]) # val
			# fig.show()
			# fig.savefig('02_outputs/fig_pso_train.png', bbox_inches='tight')

			return


def print_time(string_in):
	print(string_in, '\t -- ', datetime.datetime.now().strftime('%H:%M:%S'))
	return