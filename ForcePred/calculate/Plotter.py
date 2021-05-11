#!/usr/bin/env python

'''
This module is for plotting binned variables to see how well
sampled configurations are.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Plotter(object):
    '''
    '''

    axis_labels = 32
    tick_labels = 28

    def format(ax, x, y, xlabel, ylabel):
        ### axis labels
        ax.set_xlabel(xlabel, fontsize=Plotter.axis_labels, weight='medium')
        ax.set_ylabel(ylabel, fontsize=Plotter.axis_labels, weight='medium')
        ### ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(direction='out', length=8, width=3, colors='k',
                grid_color='k', labelsize=Plotter.tick_labels,
                bottom=True, top=True, left=True, right=True)
        ### axes labelling
        ax.tick_params(labelbottom=True, labeltop=False, 
                labelleft=True, labelright=False)
        ### axes thickness
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ### axes limits
        ax.set_xlim(np.min(x)-3, np.max(x)+3)
        ax.set_ylim(np.min(y)-3, np.max(y)+3)
        ### fix aspect ratio as square
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    def colorbar(fig, ax, sc, zlabel):
        cbar = fig.colorbar(sc,ax=ax, 
                fraction=0.046, pad=0.04, #size of legend
                #format='%.4f',
                )
        cbar.ax.tick_params(direction='out', length=6, width=3, 
                colors='k', labelsize=Plotter.tick_labels)
        cbar.ax.set_ylabel(zlabel, size=Plotter.tick_labels)
        cbar.ax.yaxis.offsetText.set_fontsize(Plotter.tick_labels-6)
        #cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.outline.set_visible(False)

    def xyz_scatter(x, y, z, xlabel, ylabel, zlabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        cm_min, cm_max = np.min(z), np.max(z)
        sc = ax.scatter(x=x, y=y, c=z, alpha=1, marker='s', 
                s=17, edgecolor='k', linewidth=0.1,
                #cmap=plt.cm.get_cmap('copper_r', 500),
                #cmap=plt.cm.get_cmap('gist_heat_r', 500),
                cmap=plt.cm.get_cmap('jet', 500),
                vmin=cm_min, vmax=cm_max)
        Plotter.format(ax, x, y, xlabel, ylabel)
        Plotter.colorbar(fig, ax, sc, zlabel)
        fig.savefig('%s' % (plot_name), 
                transparent=True, 
                bbox_inches='tight'
                )
        plt.close(plt.gcf())
