# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import stats
import os
import importlib
import DeepNetPI
import DataGen
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # avoids a warning

importlib.reload(DeepNetPI)
importlib.reload(DataGen)
importlib.reload(utils)

from DataGen import DataGenerator
from DeepNetPI import TfNetwork
from utils import *
import numpy as np
import tensorflow as tf
import datetime

start_time = datetime.datetime.now()

# inputs
type_in = 'x_cubed_gap' 	# data type to use - drunk_bow_tie x_cubed_gap ~boston
loss_type = 'qd_soft' 		# loss type to train on - qd_soft mve mse (mse=simple point prediction)
n_samples = 100		# if generating data, how many points to generate
h_size = [100]	# number of hidden units in network: [50]=layer_1 of 50, [8,4]=layer_1 of 8, layer_2 of 4
alpha = 0.05		# data points captured = (1 - alpha)
n_epoch = 150		# number epochs to train for
optim = 'adam' 		# opitimiser - SGD adam
l_rate = 0.02		# learning rate of optimiser
decay_rate=0.92		# learning rate decay
soften = 160. 		# hyper param for QD_soft
lambda_in = 15. 	# hyper param for QD_soft
sigma_in=0.3 		# initialise std dev of NN weights
is_run_test=False	# if averaging over lots of runs - turns off some prints and graphs
n_ensemble=5		# number of individual NNs in ensemble
n_bootstraps=1 		# how many boostrap resamples to perform
n_runs=20 if is_run_test else 1
is_batch=True 		# train in batches?
n_batch=100 		# batch size
lube_perc=90. 		# if model uncertainty method = perc - 50 to 100
perc_or_norm='norm' # model uncertainty method - perc norm (paper uses norm)
is_early_stop=False # stop training early (didn't use in paper)
is_bootstrap=False if n_bootstraps == 1 else True
train_prop=0.9 		# % of data to use as training

out_biases=[3.,-3.] # chose biases for output layer (for mve is overwritten to 0,1)
activation='relu' 	# NN activation fns - tanh relu

# plotting options
is_use_val=True
save_graphs=False
show_graphs=False if is_run_test else True
show_train=False if is_run_test else True
is_y_rescale=False
is_y_sort=False
is_print_info=True
var_plot=0 # lets us plot against different variables, use 0 for univariate
is_err_bars=True
is_norm_plot=False
is_boundary=True # boundary stuff ONLY works for univariate - turn off for larger
is_bound_val=False # plot validation points for boundary
is_bound_train=True # plot training points for boundary
is_bound_indiv=True # plot individual boundary estimates
is_bound_ideal=True # plot ideal boundary
is_title=True # show title w metrics on graph
bound_limit=6. # how far to plot boundary

# resampling
bootstrap_method='replace_resample' # whether to boostrap or jacknife - prop_of_data replace_resample
prop_select=0.8 # if jacknife (=prop_of_data), how much data to use each time

# other
in_ddof=1 if n_runs > 1 else 0 # this is for results over runs only

# pre calcs
if alpha == 0.05:
	n_std_devs = 1.96
elif alpha == 0.10:
	n_std_devs = 1.645
elif alpha == 0.01:
	n_std_devs = 2.575
else:
	raise Exception('ERROR unusual alpha')

results_runs = []
run=0
for run in range(0,n_runs):
	# generate data
	Gen = DataGenerator(type_in=type_in)
	X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples,seed_in=run,
		train_prop=train_prop, bound_limit=bound_limit, n_std_devs=n_std_devs)

	print('\n--- view data ---')
	Gen.ViewData(n_rows=5, hist=False, plot=False)

	X_boundary = []
	y_boundary = []
	y_pred_all = []
	X_train_orig, y_train_orig = X_train, y_train
	for b in range(0,n_bootstraps):

		# bootstrap sample
		if is_bootstrap:

			np.random.seed(b)
			if bootstrap_method=='replace_resample':
				# resample w replacement method
				id = np.random.choice(X_train_orig.shape[0], X_train_orig.shape[0], replace=True)
				X_train = X_train_orig[id]
				y_train = y_train_orig[id]

			elif bootstrap_method=='prop_of_data':
				# select x% of data each time NO resampling
				perm = np.random.permutation(X_train_orig.shape[0])
				X_train = X_train_orig[perm[:int(perm.shape[0]*prop_select)]]
				y_train = y_train_orig[perm[:int(perm.shape[0]*prop_select)]]

		i=0
		while i < n_ensemble:
			is_failed_run = False

			tf.reset_default_graph()
			sess = tf.Session()
			
			# info
			if is_print_info:
				print('\nrun number', run+1, ' of ', n_runs,
					' -- bootstrap number', b+1, ' of ', n_bootstraps,
					' -- ensemble number', i+1, ' of ', n_ensemble)

			# load network
			NN = TfNetwork(x_size=X_train.shape[1], y_size=2, h_size=h_size, 
				type_in="pred_intervals", alpha=alpha, loss_type=loss_type,
				soften=soften, lambda_in=lambda_in, sigma_in=sigma_in, activation=activation, bias_rand=False)

			# train
			NN.train(sess, X_train, y_train, X_val, y_val,
				n_epoch=n_epoch, l_rate=l_rate, decay_rate=decay_rate,
				resume_train=False, print_params=False, is_early_stop=is_early_stop,
				is_use_val=is_use_val, optim=optim, is_batch=is_batch, n_batch=n_batch,
				is_run_test=is_run_test,is_print_info=is_print_info)

			# visualise training
			if show_train:
				NN.vis_train(save_graphs, is_use_val)

			# make predictions
			y_loss, y_pred, y_metric, y_U_cap, y_U_prop, \
				y_L_cap, y_L_prop, y_all_cap, y_all_prop \
				= NN.predict(sess, X=X_val,y=y_val,in_sess=True)

			# check whether the run failed or not
			if np.abs(y_loss) > 20.:
			# if False:
				is_failed_run = True
				print('\n\n### one messed up! repeating ensemble ###')
				continue # without saving!
			else:
				i+=1 # continue to next

			# save prediction
			y_pred_all.append(y_pred)

			# predicting for boundary, need to do this for each model
			if is_boundary:
				X_boundary.append(np.linspace(start=-bound_limit,stop=bound_limit, num=500)[:,np.newaxis])
				t, y_boundary_temp, t, t, t, t, t, t, t = NN.predict(sess, X=X_boundary[i-1],
					y=np.zeros_like(X_boundary[i-1]),in_sess=True)
				y_boundary.append(y_boundary_temp)

			sess.close()

	# we may have predicted with mve or qd_soft, here we need to get estimates for 
	# upper/lower pi's AND gaussian params no matter which method we used (so can compare)
	y_pred_all = np.array(y_pred_all)

	if loss_type == 'qd_soft':
		y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
			y_pred_L = pi_to_gauss(y_pred_all, lube_perc, perc_or_norm, n_std_devs)

	elif loss_type == 'mve': # work out bounds given mu sigma
		y_pred_gauss_mid_all = y_pred_all[:,:,0]
		# occasionally may get -ves for std dev so need to do max
		y_pred_gauss_dev_all = np.sqrt(np.maximum(np.log(1.+np.exp(y_pred_all[:,:,1])),10e-6))
		y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
			y_pred_L = gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, n_std_devs)

	elif loss_type == 'mse': # as for mve but we don't know std dev so guess
		y_pred_gauss_mid_all = y_pred_all[:,:,0]
		y_pred_gauss_dev_all = np.zeros_like(y_pred_gauss_mid_all)+0.01
		y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
			y_pred_L = gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev, n_std_devs)

	# work out metrics
	y_U_cap = y_pred_U > y_val.reshape(-1)
	y_L_cap = y_pred_L < y_val.reshape(-1)
	y_all_cap = y_U_cap*y_L_cap
	PICP = np.sum(y_all_cap)/y_L_cap.shape[0]
	MPIW = np.mean(y_pred_U - y_pred_L)
	y_pred_mid = np.mean((y_pred_U,y_pred_L),axis=0)
	MSE = np.mean(np.square(Gen.scale_c*(y_pred_mid - y_val[:,0])))
	RMSE = np.sqrt(MSE)
	CWC = np_QD_loss(y_val, y_pred_L, y_pred_U, alpha, soften, lambda_in)
	neg_log_like = gauss_neg_log_like(y_val, y_pred_gauss_mid, y_pred_gauss_dev, Gen.scale_c)
	residuals = residuals = y_pred_mid - y_val[:,0]
	shapiro_W, shapiro_p = stats.shapiro(residuals[:])
	results_runs.append((PICP, MPIW, CWC, RMSE, neg_log_like, shapiro_W, shapiro_p))

	# concatenate for graphs
	title = 'PICP=' + str(round(PICP,3))\
				+ ', MPIW=' + str(round(MPIW,3)) \
				+ ', qd_loss=' + str(round(CWC,3)) \
				+ ', NLL=' + str(round(neg_log_like,3)) \
				+ ', alpha=' + str(alpha) \
				+ ', loss=' + NN.loss_type \
				+ ', data=' + type_in + ',' \
				+ '\nh_size=' + str(NN.h_size) \
				+ ', bstraps=' + str(n_bootstraps) \
				+ ', ensemb=' + str(n_ensemble) \
				+ ', RMSE=' + str(round(RMSE,3)) \
				+ ', soft=' + str(NN.soften) \
				+ ', lambda=' + str(NN.lambda_in)	

	# visualise
	if show_graphs:
		# error bars
		if is_err_bars:
			plot_err_bars(X_val, y_val, y_pred_U, y_pred_L, 
				is_y_sort, is_y_rescale, Gen.scale_c, save_graphs, 
				title, var_plot, is_title)

		# visualise boundary
		if is_boundary:
			y_bound_all=np.array(y_boundary)
			plot_boundary(y_bound_all, X_boundary, y_val, X_val, 
				y_train, X_train, loss_type,
				Gen.y_ideal_U, Gen.y_ideal_L, Gen.X_ideal, Gen.y_ideal_mean, is_bound_ideal,
				is_y_rescale, Gen.scale_c, save_graphs, 
				in_ddof, perc_or_norm, lube_perc, n_std_devs,
				is_bound_val, is_bound_train, is_bound_indiv,
				title, var_plot, is_title)

		# normal dist stuff
		if is_norm_plot:
			title = 'shapiro_W=' + str(round(shapiro_W,3)) + \
				', data=' + type_in +', loss=' + NN.loss_type + \
				', n_val=' + str(y_val.shape[0])
			fig, (ax1,ax2) = plt.subplots(2)
			ax1.set_xlabel('y_pred - y_val') # histogram
			ax1.hist(residuals,bins=30)
			ax1.set_title(title, fontsize=10)
			stats.probplot(residuals[:], plot=ax2) # QQ plot
			ax2.set_title('')
			fig.show()

# sumarise results, print for paste to excel
print('\n\nn_samples, h_size, n_epoch, l_rate, decay_rate, soften, lambda_in, sigma_in')
print(n_samples, h_size, n_epoch, l_rate, decay_rate, soften, lambda_in, sigma_in)
print('\n\ndata=',type_in, 'loss_type=',loss_type)
results_runs = np.array(results_runs)
metric_names= ['PICP', 'MPIW', 'CWC', 'RMSE', 'NLL', 'shap_W', 'shap_p']
print('runs\tboots\tensemb')
print(n_runs, '\t', n_bootstraps, '\t', n_ensemble)
print('\tavg\tstd_err\tstd_dev')
for i in range(0,len(metric_names)): 
	avg = np.mean(results_runs[:,i])
	std_dev = np.std(results_runs[:,i], ddof=in_ddof)
	std_err = std_dev/np.sqrt(n_runs)
	print(metric_names[i], '\t', round(avg,3), 
		'\t', round(std_err,3),
		'\t', round(std_dev,3))

# timing info
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))



