#!python3
import argparse
import numpy as np
from scipy.optimize import minimize
import scipy.stats
import matplotlib.pyplot as plt

def log_likelihood(t, measured_power, mu_alpha, var_alpha, mu_beta, var_beta):
	sigma_alpha = ((t ** 2) * var_alpha) 
	sigma_beta = np.ones(len(t)) * var_beta
	sigma_diag = sigma_alpha + sigma_beta
	sigma_inv = 1.0/sigma_diag
	diff = measured_power - (t * mu_alpha + mu_beta)
	
	log_sigma_diag = np.log(sigma_diag)
	total_sigma_det = np.sum(log_sigma_diag)

	data_term = (diff ** 2) * sigma_inv
	total_data_term = np.sum(data_term)
	
	if (not np.isfinite(total_sigma_det)):
		print("sigma infinite")
		total_sigma_det = 1e100
	ll = -0.5*total_sigma_det -0.5 * total_data_term
	return ll

def find_parameters(t, measured_power, mu_alpha, var_alpha, mu_beta, var_beta, BETA_VAR_MIN=None):
	BETA_VARIANCE_PRESCALE = 1.0e-3
	def optim_func(params):
		mu_alpha = params[0]
		var_alpha = params[1]
		mu_beta = params[2]
		var_beta = params[3] / BETA_VARIANCE_PRESCALE
		return -log_likelihood(t,measured_power,mu_alpha, var_alpha, mu_beta, var_beta)

	ALPHA_VAR_MIN = 1e-6
	if BETA_VAR_MIN is None :
		BETA_VAR_MIN = ALPHA_VAR_MIN
	res = minimize(optim_func, [mu_alpha, var_alpha, mu_beta, var_beta * BETA_VARIANCE_PRESCALE], 
		method='L-BFGS-B', 
		bounds=[(1e-10,None),(ALPHA_VAR_MIN,None),(None,None),(BETA_VAR_MIN * BETA_VARIANCE_PRESCALE,None)], 
		options={'gtol':1e-8, 'eps':1e-8})
	x = res.x
	x[3] = x[3] / BETA_VARIANCE_PRESCALE
	return res, x

def parameters_confidence_interval(t, mu_alpha, stddev_alpha, mu_beta, stddev_beta):
	mu = mu_alpha * t + mu_beta
	var_alpha = ((t ** 2) * stddev_alpha**2) 
	var_beta = np.ones(len(t)) * stddev_beta**2
	var = var_alpha + var_beta
	stddev = np.sqrt(var)
	min_P, max_P = scipy.stats.norm.interval(0.95, loc=mu, scale=stddev)
	return min_P, max_P

def plot_confidence_interval(t, mu_alpha, stddev_alpha, mu_beta, stddev_beta):
	min_P, max_P = parameters_confidence_interval(t, mu_alpha, stddev_alpha, mu_beta, stddev_beta)
	plt.gca().fill_between(t, max_P, min_P, color='red', alpha=0.1)
	plt.plot(t, max_P, '-', color='red', alpha=0.25)
	plt.plot(t, min_P, '-', color='red', alpha=0.25)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_file', help='CSV file with delta temperature and power consumption.')
	args = parser.parse_args()

	data = np.loadtxt(args.data_file, delimiter=',')
	T_home = data[:,0]
	P_home = data[:,1]

	INIT_ALPHA = 100.0
	INIT_BETA = -500.0
	INIT_ALPHA_VAR = 100.0
	INIT_BETA_VAR = 100.0
	res, params = find_parameters(T_home, P_home, INIT_ALPHA, INIT_ALPHA_VAR , INIT_BETA, INIT_BETA_VAR)
	mu_alpha = params[0]
	stddev_alpha = np.sqrt(params[1])
	mu_beta = params[2]
	stddev_beta = np.sqrt(params[3])

	print("Parameters : ")
	print("  mean conductance : ", mu_alpha)
	print("  conductance stddev : ", stddev_alpha)
	print("  mean offset power : ", mu_beta)
	print("  offset power stddev : ", stddev_beta)

	X_minmax = np.array([0.0, np.max(T_home)])
	Y_mean = mu_alpha * X_minmax + mu_beta

	t = np.linspace(0,np.max(T_home))

	plt.figure()	
	plot_confidence_interval(t, mu_alpha, stddev_alpha, mu_beta, stddev_beta)
	plt.plot(X_minmax, Y_mean, '-', color='black', alpha=0.75)
	plt.plot(T_home, P_home, '.', color='black')
	plt.xlabel("Temperature difference [°C]")
	plt.ylabel("Power [W]")
	plt.savefig('temperature_power_model.png')
	
	plt.figure()
	plt.plot(T_home, P_home, '.', color='black')
	plt.xlim(left=0)
	plt.ylim(bottom=0)
	plt.grid(True)
	plt.xlabel("Temperature difference [°C]")
	plt.ylabel("Power [W]")
	plt.savefig('temperature_power_data.png')

	test_t = 7.5
	test_p_mean = test_t * mu_alpha + mu_beta
	test_p_stddev = 150.0
	test_p_min = test_p_mean - 2 * test_p_stddev
	test_p_max = test_p_mean + 2 * test_p_stddev
	T_home_synthetic = np.array([test_t, test_t])
	P_home_synthetic = np.array([test_p_min, test_p_max])
	T_home_augmented = np.append(T_home, T_home_synthetic)
	P_home_augmented = np.append(P_home, P_home_synthetic)

	res, params = find_parameters(T_home_augmented, P_home_augmented, INIT_ALPHA, INIT_ALPHA_VAR , INIT_BETA, INIT_BETA_VAR)
	mu_alpha = params[0]
	stddev_alpha = np.sqrt(params[1])
	mu_beta = params[2]
	stddev_beta = np.sqrt(params[3])

	print()
	print("Parameters (augmented data ): ")
	print("  mean conductance : ", mu_alpha)
	print("  conductance stddev : ", stddev_alpha)
	print("  mean offset power : ", mu_beta)
	print("  offset power stddev : ", stddev_beta)

	Y_mean_augmented = mu_alpha * X_minmax + mu_beta

	plt.figure()
	plot_confidence_interval(t, mu_alpha, stddev_alpha, mu_beta, stddev_beta)
	plt.plot(X_minmax, Y_mean_augmented, '-', color='black', alpha=0.75)
	plt.plot(T_home, P_home, '.', color='black')
	plt.plot(T_home_synthetic, P_home_synthetic, '.', color='#5aad01')
	plt.xlabel("Temperature difference [°C]")
	plt.ylabel("Power [W]")
	plt.savefig('temperature_power_model_augmented.png')

	plt.show()

if __name__ == "__main__":
	main()