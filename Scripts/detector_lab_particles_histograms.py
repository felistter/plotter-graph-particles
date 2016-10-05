import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import numpy as np
import pandas as pd

plt.style.use('ggplot')

def save_plot(ax, name, rng, output_path):
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, 'chart_particles_{0}_{1}_{2}.png').format(name, rng[0], rng[1]))
    fig.clear()

def number(val):
    if val < 1000:
        return '%d' % val
        
    sv = str(val)
    return '$\mathregular{10^{%d}}$' % (len(sv)-2) if val % 10 == 0 else '%0.0e' % val
    
def create_pie_chart(input_path, output_path, name, rngs, voxel_size):
    df = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_scan_{0}.csv').format(name))
    df['volume(um^3)'] = df['area'] * voxel_size
    
    particles_dfs = []
    for rng in rngs:
        rng_min, rng_max = rng[0], rng[1]
        particle_df = df[(df['volume(um^3)'] > rng_min) & (df['volume(um^3)'] < rng_max)]
        vol_column = particle_df['volume(um^3)']
        particles_dfs.append(vol_column)
    
    num_particles = [len(p) for p in particles_dfs]
    sp = sum(num_particles)

    proc_particles = [n/float(sp) * 100.0 for n in num_particles]      
                      
    def get_title(v1, v2):
        return '%s $\minus$ %s $\mathregular{um^3}$' % (number(v1), number(v2))
                      
    titles = [get_title(minv, maxv) for minv,maxv in rngs]
    colors = ['#87D37C', '#65C6BB', '#1BBC9B', '#F5D76E', '#1E824C']
              
    textprops={'fontsize': 18, 'weight': 'bold', 'family': 'sans-serif'}
    pie_width = 0.5
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('equal')

    patches, texts, autotexts = ax.pie(proc_particles, \
                                       textprops=textprops, \
                                       colors=colors, \
                                       autopct='%1.1f%%', \
                                       radius=1, \
                                       pctdistance=1-pie_width/2)
    
    plt.setp(patches, \
             width=pie_width, \
             edgecolor='white')
    
    plt.legend(patches, titles, loc=(0.8,0.8))
    
    for t, p in zip(autotexts, proc_particles):
        if p < 2.0:
            pos = list(t.get_position())
            pos[0] = pos[0] + 0.5

            t.set_position(pos)
            
    plt.show()
    
    #ax.pie(proc_particles, labels=titles, autopct='%d%%', startangle=270)
    #plt.axis('equal')
              
    #series = pd.Series(np.array(proc_particles), index=titles)
    #ax = series.plot.pie(figsize=(6, 6), radius=1, pctdistance=1-width/2, legend=True)
    #plt.show()
    #save_plot(ax, name + '_pie', (rngs[0][0], rngs[-1][1]), output_path)
    

def create_plot(input_path, output_path, name, rng, voxel_size):
    df = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_scan_{0}.csv').format(name))
    df['volume(um^3)'] = df['area'] * voxel_size

    rng_min, rng_max = rng[0], rng[1]
    df = df[(df['volume(um^3)'] > rng_min) & (df['volume(um^3)'] < rng_max)]
    df = df['volume(um^3)']

    ax = df.plot(kind='hist', bins=50, color='red', xlim=(rng[0], rng[1],))
    ax.set_xlabel(r'Size of particles, $\mathregular{um^3}$')
    ax.set_ylabel("Number of particles")
    ax.set_yscale('log')
    save_plot(ax, name, rng, output_path)

def main():
    input_path = 'C:\Users\ud9751\Documents\Stats'
    output_plots_path = 'C:\Users\ud9751\Documents\Stats\Plots'
    
    voxel_size = 4 ** 3
    name = '0005'
    ranges = [(320, 10000,), (10000,100000,), (100000, 1000000,), (1000000, 10000000,)]
    
    #for rng in ranges:
    #    create_plot(input_path, output_plots_path, name, rng, voxel_size)
    
    create_pie_chart(input_path, output_plots_path, name, ranges, voxel_size)
    
if __name__ == "__main__":
    sys.exit(main())
    