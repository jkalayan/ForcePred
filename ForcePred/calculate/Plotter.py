#!/usr/bin/env python

'''
This module is for plotting binned variables to see how well
sampled configurations are.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors

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
        #ax.set_xlim(np.min(x), np.max(x))
        #ax.set_ylim(np.min(y), np.max(y))
        #ax.set_ylim(0, 103)
        #ax.set_ylim(-180, 180)
        #ax.set_xlim(-180, 180)

        ### fix aspect ratio as square
        #x0,x1 = ax.get_xlim()
        #y0,y1 = ax.get_ylim()
        #ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        ax.set_aspect('auto') #newer matplotlib version

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

    def plot_bar(x, y, xlabel, ylabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        bar = ax.bar(x, y, width=1.0, lw=2, 
                facecolor='None', edgecolor='k',)
        Plotter.format(ax, x, y, xlabel, ylabel)
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        fig.savefig('%s' % (plot_name), 
                transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())

    def hist_1d(x_list, xlabel, ylabel, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        for x in x_list:
            x_min, x_max = np.min(x), np.max(x)
            x_diff = int(x_max-x_min)
            sc = ax.hist(x=x,
                    density=True,
                    bins=50, 
                    #bins=[x_diff], 
                    #range=[[x_min, x_max]], 
                    #alpha=0.5, 
                    facecolor='None',
                    edgecolor='k',
                    linewidth=2,
                    )
        Plotter.format(ax, x, x, xlabel, ylabel)
        fig.savefig('%s' % (plot_name), 
                transparent=True, 
                bbox_inches='tight'
                )
        plt.close(plt.gcf())

    def hist_2d(x_list, y_list, cmap_list, xlabel, ylabel, 
            plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot

        for x, y, cmap in zip(x_list, y_list, cmap_list):
            x_min, x_max = np.min(x), np.max(x)
            x_diff = int(x_max-x_min)
            y_min, y_max = np.min(y), np.max(y)
            y_diff = int(y_max-y_min)

            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                        'trunc({n},{a:.2f},{b:.2f})'.format(
                        n=cmap.name, a=minval, b=maxval),
                        cmap(np.linspace(minval, maxval, n)))
                return new_cmap

            #my_cmap = plt.cm.get_cmap(cmap, 180)
            my_cmap = plt.cm.get_cmap(cmap)
            my_cmap = truncate_colormap(my_cmap, 0.3, 1)
            my_cmap.set_under(color='white', alpha='0') 
                    #set smaller values transparent
            #my_cmap.set_under((1, 1, 1, 0))

            sc = ax.hist2d(x=x, y=y,
                    bins=90,
                    #bins=[x_diff, y_diff], 
                    range=[[x_min, x_max], [y_min, y_max]], 
                    alpha=1, 
                    #marker='s', 
                    #s=17, edgecolor='k', linewidth=0.1,
                    #cmap=plt.cm.get_cmap('copper_r', 500),
                    #cmap=plt.cm.get_cmap('gist_heat_r', 500),
                    #cmap=plt.cm.get_cmap('jet', 500),
                    #cmap=plt.cm.get_cmap('binary', 500),
                    #cmap=plt.cm.get_cmap('viridis', 500),
                    #cmap=plt.cm.get_cmap(cmap, 500),
                    cmap=my_cmap,
                    cmin=0.1,
                    #vmin=0.5, vmax=1,
                    #vmin=cm_min, vmax=cm_max,
                    )
        Plotter.format(ax, x, y, xlabel, ylabel)
        fig.savefig('%s' % (plot_name), 
                #transparent=True, 
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
        Plotter.format(ax, x, y, xlabel, ylabel)
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        #lgd = Plotter.get_legend(ax, lines, label_list)
        lgd = ax.legend(
                loc='upper right', 
                #loc='lower right', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())

    def xy_scatter(x_list, y_list, label_list, color_list, 
            xlabel, ylabel, size_list, plot_name):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        lines = []
        for x, y, size, label, c in zip(x_list, y_list, size_list, 
                label_list, color_list):
            line = ax.scatter(x, y, label=label, 
                    s=size, facecolors=c, #'none', 
                    edgecolors=c, #lw=2,
                    )
            lines.append(line)
        Plotter.format(ax, x, y, xlabel, ylabel)
        #lgd = Plotter.get_legend(ax, lines, label_list)
        lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())
