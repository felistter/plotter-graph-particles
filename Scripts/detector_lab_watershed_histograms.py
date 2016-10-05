import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_plot(input_path, name, ranges, ps3):
	data_frames = []
	df_scan = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_watershed_scan_{1}.csv').format(name, name))
	df_scan['volume(um^3)'] = df_scan['area'] * ps3

	for rng in ranges:
		data_frames.append(df_scan[(df_scan['volume(um^3)'] > rng[0]) & (df_scan['volume(um^3)'] < rng[1])]['volume(um^3)'])

	fig, ax = plt.subplots()

	for df,rng in zip(data_frames, ranges):
		if not df.empty:
			ax.hist(df.values, bins=30, color='red')
			ax.set_xlabel(r'Size of particles, $\mathregular{um^3}$')
			ax.set_ylabel("Number of particles")
			ax.set_xlim([rng[0], rng[1]])
			#ax.set_yscale('log')

			plt.tight_layout()
			fig.savefig(os.path.join(input_path, 'chart_particles_watershed_{1}_{2}.png').format(name, rng[0], rng[1]))
			fig.clear()

def main():
	input_path = 'Z:\\tomo\\rshkarin\\SvetaRawData\\scan_{0}\\Analysis_temp'

	ps3 = 4 ** 3
	names = ['0010']
	#ranges = [ [(1, 1000,), (1000,10000,), (10000, 100000,), (100000, 1000000,), (1000000, 10000000,), (10000000, 100000000,), (1, 100000000,)] ]
	#ranges = [ [(10000, 5000000,), (9000, 5000000,), (8000, 5000000,), (7000, 5000000,), (6000, 5000000,), (5000, 5000000,)] ]

	ranges = [ [(100, 32000000,)] ]

	for name, rngs in zip(names, ranges):
		create_plot(input_path, name, rngs, ps3)

if __name__ == "__main__":
    sys.exit(main())
