# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import math

"""
contains some handy functions
- calculate losses from boundaries
- converting from gaussians to PIs and vice versa - incl. w ensembled estimates
- some plotting fns
"""

def print_time(string_in):
	print(string_in, '\t -- ', datetime.datetime.now().strftime('%H:%M:%S'))
	return

def np_QD_loss(y_true, y_pred_L, y_pred_U, alpha, soften = 80., lambda_in = 8.):
	"""
	manually (with np) calc the QD_hard loss
	"""
	n = y_true.shape[0]
	y_U_cap = y_pred_U > y_true.reshape(-1)
	y_L_cap = y_pred_L < y_true.reshape(-1)
	k_hard = y_U_cap*y_L_cap
	PICP = np.sum(k_hard)/n
	# in case didn't capture any need small no.
	MPIW_cap = np.sum(k_hard * (y_pred_U - y_pred_L)) / (np.sum(k_hard) + 0.001)
	loss = MPIW_cap + lambda_in * np.sqrt(n) * (max(0,(1-alpha)-PICP)**2)

	return loss


def gauss_neg_log_like(y_true, y_pred_gauss_mid, y_pred_gauss_dev, scale_c):
	"""
	return negative gaussian log likelihood
	"""
	n = y_true.shape[0]
	y_true=y_true.reshape(-1)*scale_c
	y_pred_gauss_mid=y_pred_gauss_mid*scale_c
	y_pred_gauss_dev=y_pred_gauss_dev*scale_c
	neg_log_like = -np.sum(stats.norm.logpdf(y_true.reshape(-1), loc=y_pred_gauss_mid, scale=y_pred_gauss_dev))
	neg_log_like = neg_log_like/n

	return neg_log_like


def gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, n_std_devs):
	"""
	input is individual NN estimates of mean and std dev
	1. combine into ensemble estimates of mean and std dev
	2. convert to prediction intervals
	"""
	# 1. merge to one estimate (described in paper, mixture of gaussians)
	y_pred_gauss_mid = np.mean(y_pred_gauss_mid_all,axis=0)
	y_pred_gauss_dev = np.sqrt(np.mean(np.square(y_pred_gauss_dev_all) \
		+ np.square(y_pred_gauss_mid_all),axis=0) - np.square(y_pred_gauss_mid))

	# 2. create pi's
	y_pred_U = y_pred_gauss_mid + n_std_devs*y_pred_gauss_dev
	y_pred_L = y_pred_gauss_mid - n_std_devs*y_pred_gauss_dev

	return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L

def pi_to_gauss(y_pred_all, lube_perc, perc_or_norm, n_std_devs):
	"""
	input is individual NN estimates of upper and lower bounds
	1. combine into ensemble estimates of upper and lower bounds
	2. convert to mean and std dev of gaussian
	y_pred_all is shape [no. ensemble, no. predictions, 2]
	"""
	in_ddof = 1 if y_pred_all.shape[0] > 1 else 0

	lube_perc_U = lube_perc
	lube_perc_L = 100. - lube_perc
	if perc_or_norm=='perc':
		y_pred_U = np.percentile(y_pred_all[:,:,0],q=lube_perc_U,axis=0,interpolation='linear')
		y_pred_L = np.percentile(y_pred_all[:,:,1],q=lube_perc_L,axis=0,interpolation='linear')
	elif perc_or_norm=='norm':
		# y_pred_U = np.mean(y_pred_all[:,:,0],axis=0) + 1.96*np.std(y_pred_all[:,:,0],axis=0, ddof=in_ddof)/np.sqrt(y_pred_all.shape[0])
		y_pred_U = np.mean(y_pred_all[:,:,0],axis=0) + 2.575*np.std(y_pred_all[:,:,0],axis=0, ddof=in_ddof)/np.sqrt(y_pred_all.shape[0])
		# y_pred_L = np.mean(y_pred_all[:,:,1],axis=0) - 1.96*np.std(y_pred_all[:,:,1],axis=0, ddof=in_ddof)/np.sqrt(y_pred_all.shape[0])
		y_pred_L = np.mean(y_pred_all[:,:,1],axis=0) - 2.575*np.std(y_pred_all[:,:,1],axis=0, ddof=in_ddof)/np.sqrt(y_pred_all.shape[0])

		# not STEM, just model uncert
		y_pred_U = np.mean(y_pred_all[:,:,0],axis=0) + 1.96*np.std(y_pred_all[:,:,0],axis=0, ddof=in_ddof)
		y_pred_L = np.mean(y_pred_all[:,:,1],axis=0) - 1.96*np.std(y_pred_all[:,:,1],axis=0, ddof=in_ddof)

	# need to do this before calc mid and std dev
	y_pred_U_temp = np.maximum(y_pred_U, y_pred_L)
	y_pred_L = np.minimum(y_pred_U, y_pred_L)
	y_pred_U = y_pred_U_temp

	y_pred_gauss_mid = np.mean((y_pred_U,y_pred_L),axis=0)
	y_pred_gauss_dev = (y_pred_U - y_pred_gauss_mid) / n_std_devs

	return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L


def plot_err_bars(X_val, y_val, y_pred_U, y_pred_L, 
	is_y_sort, is_y_rescale, scale_c_in, save_graphs, title='',
	var_plot=0, is_title=True):
	"""
	plot the error bar plot
	"""

	fig, ax = plt.subplots(1)

	# whether to rescale or not
	if is_y_rescale:
		scale_c = scale_c_in # times output by constant to rescale if want
		ax.set_ylabel('y')
	else:
		scale_c = 1.0
		ax.set_ylabel('y normalised')

	# error bars
	if is_y_sort:
		# order small to large
		id = np.argsort(y_val,axis=0)[:,0]
		ax.errorbar(np.arange(0,y_val.shape[0]), scale_c*y_val[id,0], 
			yerr=[scale_c*y_val[id,0]-scale_c*y_pred_L[id], 
			scale_c*y_pred_U[id]-scale_c*y_val[id,0]], 
			ecolor='r', alpha=0.6, elinewidth=0.6, capsize=2.0, capthick=1., fmt='none', label='PIs')
		ax.scatter(np.arange(0,y_val.shape[0]),scale_c*y_val[id],c='b',s=2.0)
		ax.set_xlabel('Samples ordered by y value')
	else:
		# plot as is
		ax.errorbar(X_val[:,var_plot], scale_c*y_val[:,0], 
			yerr=[scale_c*y_val[:,0]-scale_c*y_pred_L, 
			scale_c*y_pred_U-scale_c*y_val[:,0]], 
			ecolor='r', alpha=0.6, elinewidth=0.6, capsize=2.0, capthick=1., fmt='none', label='PIs')
		ax.scatter(X_val[:,var_plot],scale_c*y_val,c='b',s=2.0)
		ax.set_xlabel('X')

	if is_title:
		ax.set_title(title, fontsize=10)

	ax.legend(loc='upper left')
	if save_graphs:
		fig.savefig('02_outputs/pi_err_bar.png', bbox_inches='tight')
	else:
		fig.show()

	return


def plot_boundary(y_bound_all, X_boundary, y_val, X_val, 
			y_train, X_train, loss_type,
			y_ideal_U, y_ideal_L, X_ideal, y_ideal_mean, is_bound_ideal,
			is_y_rescale, scale_c_in, save_graphs, 
			in_ddof, perc_or_norm, lube_perc, n_std_devs,
			is_bound_val, is_bound_train, is_bound_indiv,
			title='', var_plot=0, is_title=True):
	"""
	plot boundary
	"""
	fig, ax = plt.subplots(1)

	# whether to rescale or not
	if is_y_rescale:
		scale_c = scale_c_in # times output by constant to rescale if want
		ax.set_ylabel('y')
	else:
		scale_c = 1.0
		ax.set_ylabel('y')

	if loss_type == 'qd_soft' or loss_type == 'umae' or loss_type=='qd_hard':
		y_bound_gauss_mid, y_bound_gauss_dev, y_bound_U, \
			y_bound_L = pi_to_gauss(y_bound_all, lube_perc, perc_or_norm, n_std_devs)

	elif loss_type == 'gauss_like': # work out bounds given mu sigma
		y_bound_gauss_mid_all = y_bound_all[:,:,0]
		y_bound_gauss_dev_all = np.sqrt(np.maximum(np.log(1.+np.exp(y_bound_all[:,:,1])),10e-6))
		y_bound_gauss_mid, y_bound_gauss_dev, y_bound_U, \
			y_bound_L = gauss_to_pi(y_bound_gauss_mid_all, y_bound_gauss_dev_all, n_std_devs)


	# if have individual estimates !!! need to convert gauss -> multiple if want this
	if is_bound_indiv:
		if loss_type == 'qd_soft' or loss_type == 'umae':
			for j in range(0, y_bound_all.shape[0]): # for each NN
				ax.plot(X_boundary[j][:,var_plot], scale_c*y_bound_all[j][:,0],
					c='g',alpha=1.0,linestyle='--',linewidth=0.2)
				ax.plot(X_boundary[j][:,var_plot], scale_c*y_bound_all[j][:,1],
					c='g',alpha=1.0,linestyle='--',linewidth=0.2)

			# dummy lines so can get labels out
			ax.plot(X_boundary[j][0,var_plot], scale_c*y_bound_all[j][0,0],
				c='0.6',alpha=1.,linestyle=':',linewidth=1., label='Model uncertainty')
			# ax.plot(X_boundary[j][0,var_plot], scale_c*y_bound_all[j][0,0],
			# 	c='b',alpha=0.4,linewidth=0.6, linestyle='-', label='true data fn')
			ax.plot(X_boundary[j][0,var_plot], scale_c*y_bound_all[j][0,0],
				c='g',alpha=1.0,linestyle='--',linewidth=0.5 , label='Indiv. boundaries')
			
			# plot std dev of estimates
			ax2 = ax.twinx()
			# ax2.plot(X_boundary[0][:,var_plot], scale_c*np.std(y_bound_all,axis=0)[:,0],
			# 	c='0.5',alpha=1.,linestyle=':',linewidth=1., label='y_U uncert')
			# ax2.plot(X_boundary[0][:,var_plot], scale_c*np.std(y_bound_all,axis=0)[:,1],
			# 	c='0.5',alpha=1.,linestyle=':',linewidth=1., label='y_L uncert')
			model_uncert = np.std(y_bound_all,axis=0)
			model_uncert = np.mean(model_uncert,axis=1) # combine U and L bounds into one
			ax2.plot(X_boundary[0][:,var_plot], scale_c*model_uncert,
			 	c='0.3',alpha=1.,linestyle=':',linewidth=1., label='Model uncertainty')
			ax2.set_ylabel('Estimated model uncertainty')

		elif loss_type == 'gauss_like':
			for j in range(0, y_bound_all.shape[0]): # for each NN
				y_b_j_mid, y_b_j_dev, y_b_j_U, y_b_j_L = gauss_to_pi(y_bound_gauss_mid_all[j,][np.newaxis], 
					y_bound_gauss_dev_all[j,][np.newaxis], n_std_devs)
				ax.plot(X_boundary[j][:,var_plot], scale_c*y_b_j_U,
					c='g',alpha=0.3,linestyle='--',linewidth=0.5)# , label='y_U')
				ax.plot(X_boundary[j][:,var_plot], scale_c*y_b_j_L,
					c='g',alpha=0.3,linestyle='--',linewidth=0.5)#, label='y_L')


	# fill
	ax.fill_between(X_boundary[0][:,var_plot], 
		scale_c*y_bound_U, scale_c*y_bound_L,
		color='0.9', alpha=1.0)

	# fill outlines
	ax.plot(X_boundary[0][:,var_plot],scale_c*y_bound_U,
		c='k',alpha=1.0,linewidth=0.7, label='Ensemble boundary')
	ax.plot(X_boundary[0][:,var_plot],scale_c*y_bound_L,
		c='k',alpha=1.0,linewidth=0.7)

	# scatter plot of data points
	if is_bound_val:
		ax.scatter(X_val[:,var_plot],scale_c*y_val,c='k',s=0.8, alpha=0.5)
	if is_bound_train:
		ax.scatter(X_train[:,var_plot],scale_c*y_train,c='r',s=0.8)
		# ax.scatter(X_train[:,var_plot],scale_c*y_train,c='m',s=10.0,marker='x')

	if is_bound_ideal:
		# outline of ideal area
		# ax.plot(X_ideal,scale_c*y_ideal_U,
		# 	c='b',alpha=0.4,linewidth=0.6, linestyle='--')
		# ax.plot(X_ideal,scale_c*y_ideal_L,
		# 	c='b',alpha=0.4,linewidth=0.6, linestyle='--')
		ax.plot(X_ideal,scale_c*y_ideal_mean,
			c='b',alpha=0.4,linewidth=0.6, linestyle='-', label='True data fn')

	if is_title:
		ax.set_title(title, fontsize=10)

	leg = ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.0))
	leg.get_frame().set_alpha(1.)

	ax.set_xlabel('x')

	if save_graphs:
		# fig.savefig('02_outputs/pi_boundary.png', bbox_inches='tight')
		fig.savefig('02_outputs/pi_b_ens_last_9.eps', format='eps', dpi=1000, bbox_inches='tight')

	fig.show()

	return


