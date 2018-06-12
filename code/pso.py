
import os
import importlib
import matplotlib.pyplot as plt
import DeepNetPI
import DataGen
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # avoids meaningless warning

importlib.reload(DeepNetPI)
importlib.reload(DataGen)
importlib.reload(utils)

from DataGen import DataGenerator
from DeepNetPI import TfNetwork
from DeepNetPI import swarm
import tensorflow as tf
import numpy as np
from utils import *

# inputs
type_in = "drunk_bow_tie"
loss_type = 'qd_hard'
n_samples = 100
h_size = [50]
alpha = 0.05
soften = 160.
lambda_in = 15.
sigma_in = 0.6
particle_n = 40
n_epoch = 800
params_in = [[0.721, 1.193, 1.]] # v_prev, v_global & v_local, v_max

out_biases=[3.,-3.]
activation='relu' # tanh relu

# [0.60, 1.4, 9.] 0.59
# [0.55, 1.4, 10.] sigma=0.5 0.58
# [0.55, 1.5, 10.0] sigma=0.6  0.61

#plot options
is_bound_ideal=True
is_y_rescale=True
save_graphs=False
in_ddof=1
perc_or_norm='norm'
lube_perc=90.
n_std_devs=1.96
is_bound_val=False
is_bound_train=True
is_bound_indiv=False
is_title=False
var_plot=0
bound_limit = 2.

# generate data
Gen = DataGenerator(type_in=type_in)
X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples, bound_limit=bound_limit)
print('\n--- view data ---')
Gen.ViewData(n_rows=5, hist=False, plot=False)


for i in range(0, len(params_in)):

	# start session
	tf.reset_default_graph()
	sess = tf.Session()
	# sess.run(tf.global_variables_initializer())
	# sess = tf.InteractiveSession()
	# tf.global_variables_initializer().run() # init variables

	print('--- loop:', i, '---\nparams: ', params_in[i])

	# initialise and train
	S = swarm(X_train, y_train, X_val, y_val, 
		x_size=X_train.shape[1], y_size=2, h_size=h_size, 
		type_in="pred_intervals", sess=sess, alpha=alpha, loss_type=loss_type,
		soften=soften, lambda_in=lambda_in, sigma_in=sigma_in,
		particle_n=particle_n, 
		v_prev=params_in[i][0],v_global=params_in[i][1], 
		v_local=params_in[i][1], v_max=params_in[i][2],
		n_epoch=n_epoch, activation=activation)

	# v_prev=0.721,v_global=1.193, v_local=1.193, v_max=2.0

	S.vis_train()


	# make predictions
	y_loss, y_pred, y_metric, y_U_cap, y_U_prop, \
		y_L_cap, y_L_prop, y_all_cap, y_all_prop \
		= S.best_NN.predict(sess, X=X_val,y=y_val,in_sess=True)

	# boundary stuff
	X_boundary=[]
	y_boundary=[]
	X_boundary.append(np.linspace(start=-bound_limit,stop=bound_limit, num=500)[:,np.newaxis])
	t, y_boundary_temp, t, t, t, t, t, t, t = S.best_NN.predict(sess, X=X_boundary[0],
		y=np.zeros_like(X_boundary[0]),in_sess=True)
	y_boundary.append(y_boundary_temp)

	sess.close()

	# print summary
	print('y_metric: U capt., L capt., PICP, MPIW:',y_metric)
	print('y_loss:',y_loss)

	title = 'loss=' + str(round(S.best_global_loss,4)) \
		+ ', loss_type=' + S.loss_type \
		+ ', v_prev=' + str(S.v_prev) \
		+ ', \nv_global=' + str(S.v_global) \
		+ ', v_max=' + str(S.v_max) \
		+ ', particle_n=' + str(S.particle_n)

	# visualise prediction intervals
	fig, ax = plt.subplots(1)
	i=0 # use 0 for univariate
	ax.scatter(X_val[:,i],y_val,c='b',s=2.0)
	ax.scatter(X_val[:,i], y_pred[:,0],c='g',alpha=0.5,s=1.0) # y_U
	ax.scatter(X_val[:,i], y_pred[:,1],c='r',alpha=0.5,s=1.0) # y_L
	# plt.fill_between(X_val.reshape(-1),y_pred[:,0],y_pred[:,1])
	ax.set_ylabel('y')
	ax.set_xlabel('X')
	ax.set_title(title)
	# ax.set_title('info here')
	fig.show()
	if save_graphs:
		fig.savefig('02_outputs/fig_pso_pi.png', bbox_inches='tight')

	# boundary plot
	y_bound_all=np.array(y_boundary)
	plot_boundary(y_bound_all, X_boundary, y_val, X_val, 
		y_train, X_train, loss_type,
		Gen.y_ideal_U, Gen.y_ideal_L, Gen.X_ideal, Gen.y_ideal_mean, is_bound_ideal,
		is_y_rescale, Gen.scale_c, save_graphs, 
		in_ddof, perc_or_norm, lube_perc, n_std_devs,
		is_bound_val, is_bound_train, is_bound_indiv,
		title, var_plot, is_title)

