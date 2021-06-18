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
        #ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        #ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis='both', which='both', direction='out', 
                length=8, width=3, colors='k',
                grid_color='k', labelsize=Plotter.tick_labels,
                bottom=True, top=True, left=True, right=True)
        ### axes labelling
        ax.tick_params(labelbottom=True, labeltop=False, 
                labelleft=True, labelright=False)
        ### axes thickness
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ### axes limits
        #ax.set_xlim(np.min(x)-3, np.max(x)+3)
        #ax.set_ylim(np.min(y)-3, np.max(y)+3)
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

    def get_legend(ax, lines, labels):
        lgd_x = 1
        lgd_y = 0
        lgd = ax.legend(lines, labels,
                #loc='center left',
                #bbox_to_anchor=(lgd_x, lgd_y),
                fontsize=Plotter.tick_labels+4)
        lgd.get_frame().set_alpha(0)
        return lgd

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

    def hist_2d(x, y, xlabel, ylabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        x_min, x_max = np.min(x), np.max(x)
        x_diff = int(x_max-x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_diff = int(y_max-y_min)
        sc = ax.hist2d(x=x, y=y, bins=[x_diff, y_diff], 
                range=[[x_min, x_max], [y_min, y_max]], 
                alpha=1, 
                #marker='s', 
                #s=17, edgecolor='k', linewidth=0.1,
                #cmap=plt.cm.get_cmap('copper_r', 500),
                #cmap=plt.cm.get_cmap('gist_heat_r', 500),
                #cmap=plt.cm.get_cmap('jet', 500),
                #cmap=plt.cm.get_cmap('binary', 500),
                cmap=plt.cm.get_cmap('viridis', 500),
                #vmin=cm_min, vmax=cm_max
                )
        Plotter.format(ax, x, y, xlabel, ylabel)
        fig.savefig('%s' % (plot_name), 
                transparent=True, 
                bbox_inches='tight'
                )
        plt.close(plt.gcf())

    def plot_2d(x_list, y_list, label_list, xlabel, ylabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        lines = []
        for x, y, label in zip(x_list, y_list, label_list):
            line = ax.plot(x, y, lw=3, label=label)
            lines.append(line)
        ax.set_xscale('log')
        Plotter.format(ax, x, y, xlabel, ylabel)
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        #lgd = Plotter.get_legend(ax, lines, label_list)
        lgd = ax.legend(loc='lower right', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())


    def xy_scatter(x_list, y_list, label_list, color_list, 
            xlabel, ylabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        lines = []
        for x, y, label, c in zip(x_list, y_list, label_list, color_list):
            line = ax.scatter(x, y, label=label, 
                    s=2, facecolors='none', edgecolors=c
                    )
            lines.append(line)
        Plotter.format(ax, x, y, xlabel, ylabel)
        #lgd = Plotter.get_legend(ax, lines, label_list)
        lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())
