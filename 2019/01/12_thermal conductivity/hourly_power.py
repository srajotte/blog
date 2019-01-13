#!python3

import numpy as np
import matplotlib.pyplot as plt

def set_aspect(ax, ratio):
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def main():
	hour = np.arange(24)
	power = np.array([1.24,1.82,1.32,1.57,1.72,1.84,1.99,5.3,6.49,3.98,3.35,2.98,2.09,4.76,2.51,2.46,3.47,6.02,7.52,7.08,5,4.08,1.63,0.41])
	plt.grid(True)
	plt.plot(hour, power, '.:')
	plt.xticks(np.arange(0, 23+1, 2.0))
	plt.xlabel("Time of day [hour]")
	plt.ylabel("Average electric power [kW]")
	set_aspect(plt.gca(), 0.5)
	plt.savefig('hourly_power.png', dpi=300, bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	main()