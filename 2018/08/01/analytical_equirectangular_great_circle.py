#!python3

import numpy as np
import matplotlib.pyplot as plt

def equirectangular_great_circle(longitude_array, azimuth, inclination):
	""" longitude_array : array of longitude samples in degrees.
		azimuth : azimuth angle in degrees.
	    inclination : inclination angle in degrees. """
	return np.rad2deg(np.arctan(np.tan(np.deg2rad(inclination)) * (np.sin(np.deg2rad(longitude_array) - np.deg2rad(azimuth)))))

def set_equirectangluar_plot():
	plt.xlim(-180, 180)
	plt.ylim(-90, 90)
	plt.xticks(np.arange(-180.0,181.0, step=45.0))
	plt.yticks(np.arange(-90.0,91.0, step=30.0))
	plt.xlabel('Longitude [degree]')
	plt.ylabel('Lattitude [degree]')
	plt.grid()

def plot_gc_list(x, gc, legend_title=None, labels=None):
	for l, y in zip(labels, gc):
		plt.plot(x, y, label=str(l) + '°')
	set_equirectangluar_plot()
	plt.legend(ncol=2, title=legend_title)

def main():
	N_samples = 1001
	longitude_samples = np.rad2deg(np.linspace(-np.pi, np.pi, N_samples))

	inclinations = list(range(0, 90+1, 15)) + [85, 89]
	inclinations.sort()
	azimuth = 0
	gc = list(equirectangular_great_circle(longitude_samples, azimuth, i) for i in inclinations)

	plt.figure()
	plot_gc_list(longitude_samples, gc, 'inclination', inclinations)
	plt.savefig('equirectangular_great_circle_inclination.png', dpi=200)

	azimuths = list(range(-180, 180+1, 45))
	inclinations.sort()
	inclination = 85
	gc = list(equirectangular_great_circle(longitude_samples, a, inclination) for a in azimuths)

	plt.figure()
	plot_gc_list(longitude_samples, gc, 'azimuth', azimuths)
	plt.savefig('equirectangular_great_circle_azimuth.png', dpi=200)

	azimuth = -130
	inclination = 51.6
	gc = equirectangular_great_circle(longitude_samples, azimuth, inclination)

	plt.figure()
	img = plt.imread('land_ocean_ice_2048.jpg')
	plt.imshow(img, zorder=0, extent=[-180, 180, -90, 90], alpha=0.75)
	plt.plot(longitude_samples,gc, color='red')
	set_equirectangluar_plot()
	plt.savefig('equirectangular_ISS.png', dpi=200)
	
	plt.show()

if __name__ == '__main__':
	main()