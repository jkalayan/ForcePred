#!/usr/bin/env python

'''
This module is for plotting multiple plots in oone file.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde #density
from matplotlib import ticker

class MultiPlotter(object):
    """
    """

    axis_labels = 10 #32 # fontsize 10 is matplotlib default
    tick_labels = axis_labels-2 #28

    def format(ax, x, y, xlabel, ylabel):
        ### axis labels
        # ax.set_xlabel(xlabel, fontsize=MultiPlotter.axis_labels)
        # ax.set_ylabel(ylabel, fontsize=MultiPlotter.axis_labels)

        ### ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.tick_params(axis='both', which='both', direction='out', 
                length=2, colors='k',
                grid_color='k', labelsize=MultiPlotter.tick_labels,
                #bottom=True, top=True, left=True, right=True
                )
        
        ### axes labelling
        #ax.tick_params(labelbottom=True, labeltop=False, 
                #labelleft=True, labelright=False)

        ### axes thickness
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax.spines[axis].set_linewidth(3)

        ### axes limits
        #ax.set_xlim(np.min(x), np.max(x))
        #ax.set_ylim(np.min(y), np.max(y))
        #ax.set_ylim(0, 103)
        #ax.set_ylim(-180, 180)
        #ax.set_xlim(-180, 180)

        ### fix aspect ratio as square
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        # ax.set_aspect('auto') #newer matplotlib version?

    def colorbar(fig, ax, sc, zlabel):
        cbar = fig.colorbar(sc,
                cax=ax, 
                #ax=ax, 
                fraction=0.046, pad=0.04, #size of legend
                #format='%.2f',
                )
        cbar.ax.tick_params(direction='out', length=6, width=3, 
                colors='k', labelsize=MultiPlotter.tick_labels)
        cbar.ax.set_ylabel(zlabel, size=MultiPlotter.tick_labels)
        cbar.ax.yaxis.offsetText.set_fontsize(MultiPlotter.tick_labels-6)
        #cbar.ax.locator_params(nbins=3) #relates to ticks
        #cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.outline.set_visible(False)

    def get_legend(ax, lines, labels):
        lgd_x = 1
        lgd_y = 0
        lgd = ax.legend(lines, labels,
                #loc='center left',
                #bbox_to_anchor=(lgd_x, lgd_y),
                fontsize=MultiPlotter.tick_labels+4)
        lgd.get_frame().set_alpha(0)
        return lgd
    
    def setup_plot(name_list):
        """
        """
        fig = plt.figure(layout="compressed")
        layout_list = [name_list[:len(name_list)//2], 
                       name_list[len(name_list)//2:]]
        ax_dict = fig.subplot_mosaic(layout_list)
        return fig, ax_dict

    def save_plot(fig, plot_name):
        """
        """
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())

    def violin(name_list, data_list):
        """
        """
        fig, ax_dict = MultiPlotter.setup_plot(name_list)
        all_dihs = []
        for i in range(len(name_list)):
            name = name_list[i]
            data = data_list[i]
            x = data.phis.T[0]
            all_dihs.append(x)
            x_min, x_max = np.min(x), np.max(x)
            x_diff = int(x_max-x_min)
            # sc = ax_dict[name].scatter(x=x, y=x)
            # sc = ax_dict[name].hist(x)
            # sc = ax_dict[name].violinplot(x)
            sc = ax_dict[name].hist(x=x,
                    density=True,
                    #bins=50, 
                    # bins=[x_diff], 
                    # range=[x_min, x_max], 
                    range=[-180, 180], 
                    #alpha=0.5, 
                    # facecolor='None',
                    # edgecolor='k', #c,
                    # linewidth=2,
                    )
            MultiPlotter.format(ax_dict[name], x, x, xlabel="x", ylabel="y")
        MultiPlotter.save_plot(fig, "test.pdf")

        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        sc = ax.violinplot(all_dihs)

        def set_axis_style(ax, labels):
            ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            ax.set_xlabel('Sample name')

        set_axis_style(ax, name_list)
        MultiPlotter.save_plot(fig, "2test.pdf")
