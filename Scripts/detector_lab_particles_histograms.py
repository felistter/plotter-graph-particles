import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab


COLORS = ['#87D37C', '#65C6BB', '#1BBC9B', '#F5D76E', '#1E824C']

plt.style.use('ggplot')

def create_folder_with_path(path, folder_name):
    path = os.path.join(path, folder_name)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path

def create_distribution_std(input_path, name, voxel_size, rng):
    df = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_scan_{0}.csv').format(name))
    df['volume(um^3)'] = df['area'] * voxel_size
    
    particle_df = df[(df['volume(um^3)'] > rng[0]) & (df['volume(um^3)'] < rng[1])]
    particle_df = particle_df['volume(um^3)']
    std = particle_df.std()
    mean = particle_df.mean()
                              
    n, bins, patches = plt.hist(particle_df.values, bins=50, normed=True, alpha=0.5)
    
    y = mlab.normpdf(bins, mean, std)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Particles size')
    plt.ylabel('Deviation')
    plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    plt.show()

def save_plot(ax, name, rng, output_path, logscale=False):
    fig = ax.get_figure()
    #plt.tight_layout()
    fig.savefig(os.path.join(output_path, 'chart_particles_{0}_{1}_{2}_{3}.png').format(name, rng[0], rng[1], 'log' if logscale else 'ariphm'))
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
              
    textprops={'fontsize': 22, 'weight': 'light', 'family': 'sans-serif'}
    pie_width = 0.5
    fig, ax = plt.subplots(figsize=(8.5,8))
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
    
    plt.legend(patches, titles, loc=(0.8,0.8), fontsize=16)
    
    for t, p in zip(autotexts, proc_particles):
        if p < 2.0:
            pos = list(t.get_position())
            pos[0] = pos[0] + 0.45

            t.set_position(pos)
            
    #plt.show()
    
    #ax.pie(proc_particles, labels=titles, autopct='%d%%', startangle=270)
    #plt.axis('equal')
              
    #series = pd.Series(np.array(proc_particles), index=titles)
    #ax = series.plot.pie(figsize=(6, 6), radius=1, pctdistance=1-width/2, legend=True)
    #plt.show()
    plt.subplots_adjust(left=-0.08, right=0.9, top=1, bottom=-0.08)
    
    output_path = create_folder_with_path(output_path, 'Pie_charts')
        
    save_plot(ax, name + '_pie', (rngs[0][0], rngs[-1][1]), output_path)

def create_plot_stack(input_path, output_path, names, rngs, voxel_size, colors, logscale=False):
    for name, rgn_collection, color in zip(names, rngs, colors):
        df = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_scan_{0}.csv').format(name))
        df['volume(um^3)'] = df['area'] * voxel_size
           
        for rng in rgn_collection:
            rng_min, rng_max = rng[0], rng[1]
            df = df[(df['volume(um^3)'] > rng_min) & (df['volume(um^3)'] < rng_max)]
            df = df['volume(um^3)']
        
            ax = df.plot(kind='hist', bins=50, color=color, xlim=(rng[0], rng[1],), figsize=(14,10), fontsize=16)
            ax.set_xlabel(r'Size of particles, $\mathregular{um^3}$', color='black', fontsize=16, labelpad=20)
            ax.set_ylabel("Number of particles", color='black', fontsize=16, labelpad=20)
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            if logscale: 
                ax.set_yscale('log')
       
    output_path = create_folder_with_path(output_path, ('Hists_ariphm_charts' if not logscale else 'Hists_log_charts') + '_stack')
       
    save_plot(ax, '_'.join(names), (rngs[0][0][0], rngs[-1][0][1]), output_path, logscale=logscale)
    

def create_plot(input_path, output_path, name, rng, voxel_size, logscale=False):
    df = pd.DataFrame.from_csv(os.path.join(input_path, 'particles_stats_scan_{0}.csv').format(name))
    df['volume(um^3)'] = df['area'] * voxel_size

    rng_min, rng_max = rng[0], rng[1]
    df = df[(df['volume(um^3)'] > rng_min) & (df['volume(um^3)'] < rng_max)]
    df = df['volume(um^3)']

    ax = df.plot(kind='hist', bins=50, color='#1BBC9B', xlim=(rng[0], rng[1],), figsize=(14,10), fontsize=16)
    ax.set_xlabel(r'Size of particles, $\mathregular{um^3}$', color='black', fontsize=16, labelpad=20)
    ax.set_ylabel("Number of particles", color='black', fontsize=16, labelpad=20)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    if logscale: 
        ax.set_yscale('log')
       
    output_path = create_folder_with_path(output_path, 'Hists_ariphm_charts' if not logscale else 'Hists_log_charts')
       
    save_plot(ax, name, rng, output_path, logscale=logscale)

def main():
    input_path = '..'
    output_plots_path = os.path.join('..', 'Plots')
    
    voxel_size = 3.7 ** 3
    name = '0005'
                  
    # Arihm plots
    ranges_hist = [(100, 32000000,)]                  
    #ranges_hist = [(320, 10000,), (10000,100000,), (100000, 1000000,), (1000000, 10000000,), (320,10000000)]
    for rng in ranges_hist:
        create_plot(input_path, output_plots_path, name, rng, voxel_size)
    
    # Log plots
    ranges_loghist = [(320, 10000000,)]
    for rng in ranges_loghist:
        create_plot(input_path, output_plots_path, name, rng, voxel_size, logscale=True)
        
    # Pie chart
    ranges_pie = [(320, 10000,), (10000,100000,), (100000, 1000000,), (1000000, 10000000,)]
    create_pie_chart(input_path, output_plots_path, name, ranges_pie, voxel_size)
    
    # Histogram + Gaussian dist
    #create_distribution_std (input_path, name, voxel_size, (5000, 10000))
    
    # Staked histogram
    #names = ['0005', '0007', '0008', '0010']
    #ranges_hist = [[(100, 32000000,)],[(100, 32000000,)],[(100, 32000000,)], [(100, 32000000,)]] 
    #create_plot_stack(input_path, output_plots_path, names, ranges_hist, voxel_size, COLORS[:4], logscale=True)
    
if __name__ == "__main__":
    sys.exit(main())
    