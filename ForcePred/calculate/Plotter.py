#!/usr/bin/env python

'''
This module is for plotting binned variables to see how well
sampled configurations are.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde #density
from matplotlib import ticker
from scipy import interpolate

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
        #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis='both', which='both', direction='out', 
                length=8, width=3, colors='k',
                grid_color='k', labelsize=Plotter.tick_labels,
                #bottom=True, top=True, left=True, right=True
                )
        ### axes labelling
        #ax.tick_params(labelbottom=True, labeltop=False, 
                #labelleft=True, labelright=False)
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
        cbar = fig.colorbar(sc,
                cax=ax, 
                #ax=ax, 
                fraction=0.046, pad=0.04, #size of legend
                #format='%.2f',
                )
        cbar.ax.tick_params(direction='out', length=6, width=3, 
                colors='k', labelsize=Plotter.tick_labels)
        cbar.ax.set_ylabel(zlabel, size=Plotter.tick_labels)
        cbar.ax.yaxis.offsetText.set_fontsize(Plotter.tick_labels-6)
        #cbar.ax.locator_params(nbins=3) #relates to ticks
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
                #transparent=True, 
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
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())

    def hist_1d(x_list, xlabel, ylabel, plot_name, color_list=['k']):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        for x, c in zip(x_list, color_list):
            x_min, x_max = np.min(x), np.max(x)
            x_diff = int(x_max-x_min)
            sc = ax.hist(x=x,
                    density=True,
                    bins=50, 
                    #bins=[x_diff], 
                    #range=[[x_min, x_max]], 
                    #alpha=0.5, 
                    facecolor='None',
                    edgecolor=c, #'k',
                    linewidth=2,
                    )
        Plotter.format(ax, x, x, xlabel, ylabel)
        fig.savefig('%s' % (plot_name), 
                #transparent=True, 
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
                    #range=[[x_min, x_max], [y_min, y_max]], 
                    #range=[[-180, 180], [-50, 50]], 
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
        #fig.colorbar(sc[3], ax=ax)
        cb_ax = fig.add_axes([0.95, 0.12, 0.05, 0.75])
        Plotter.colorbar(fig, cb_ax, sc[3], '$N_\mathrm{structures}$')
        #cbar = fig.colorbar(sc[3], cax=cb_ax)
        #cbar.ax.tick_params(labelsize=Plotter.tick_labels)
        #cbar.outline.set_linewidth(3)
        fig.savefig('%s' % (plot_name), 
                #transparent=True, 
                bbox_inches='tight'
                )
        plt.close(plt.gcf())

    def plot_2d(x_list, y_list, label_list, xlabel, ylabel, plot_name, 
            log=False):
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        lines = []
        for x, y, label in zip(x_list, y_list, label_list):
            line = ax.plot(x, y, lw=3, label=label)
            lines.append(line)
        Plotter.format(ax, x, y, xlabel, ylabel)
        if log:
            ax.set_xscale('log')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        ax.grid(color='grey', linestyle='--', which='major', 
                axis='both', linewidth=1)
        #lgd = Plotter.get_legend(ax, lines, label_list)
        #'''
        lgd = ax.legend(
                loc='upper left', bbox_to_anchor=(1.1, 1), 
                #loc='lower right', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        #'''
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())


    def twinx_plot(x_list, y_list, label_list, color_list, ls_list, 
            marker_list, xlabel, ylabel, ylabel2, size_list, axis_list, 
            func, plot_name):
        '''Only for two differing scales'''
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        ax2 = ax.twinx()

        lines = []
        lines2 = []
        count = 0
        for x, y, size, a, label, c, ls, m in zip(x_list, y_list, size_list, 
                axis_list, label_list, color_list, ls_list, marker_list):
            inds2 = x.argsort()
            x = x[inds2]
            y = y[inds2]
            print('***', len(x), len(y))
            print('---', x, y)
            print()
            if a == 1:
                line = ax.scatter(x, y, label=label, 
                        s=size, facecolors='none', #c, #
                        edgecolors=c, marker=m, lw=2,
                        )
            if a == 2:
                line = ax2.scatter(x, y, label=label, 
                        s=size, facecolors=c, #'none', #
                        edgecolors=c, marker=m, lw=3,
                        )
            lines.append(line)

            x2, inds = np.unique(x, return_index=True)
            x_unique = x[inds]
            y_unique = y[inds]
            xnew = np.linspace(x_unique.min(), x_unique.max(), 1000)
            f_smooth = interp1d(x_unique, y_unique, kind=2)
            poly = np.polyfit(x_unique,y_unique,5)
            poly_y = np.poly1d(poly)(xnew)
            if func == 'f':
                curve = f_smooth(xnew)
                #poly = np.polyfit(x_unique,y_unique,27)
                #poly_y = np.poly1d(poly)(xnew)
                #curve = poly_y
            if func  == 'p':
                curve = poly_y
            if a == 1:
                line2 = ax.plot(xnew, 
                        #f_smooth(xnew), 
                        curve, 
                        c=c, lw=2, ls=ls,
                        #label=label
                        )
            if a == 2:
                line2 = ax2.plot(xnew, 
                        #f_smooth(xnew), 
                        curve, 
                        c=c, lw=3, ls=ls,
                        #label=label
                        )
            lines2.append(line2)
            count += 1


        #ax2.set_ylabel(ylabel2, fontsize=Plotter.axis_labels, 
                #weight='medium')

        Plotter.format(ax2, x, y, xlabel, ylabel2)
        Plotter.format(ax, x, y, xlabel, ylabel)
        #ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax2.yaxis.offsetText.set_fontsize(Plotter.tick_labels)
        ax.grid(color='grey', linestyle='--', which='major', 
                axis='both', linewidth=1)
        labs = [l.get_label() for l in lines]
        lgd = ax.legend(lines, labs, bbox_to_anchor=(1.1, 1), 
                loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())


    def error_bars_plot(x_list, y_list, error_list, label_list, 
            color_list, ls_list, 
            marker_list, xlabel, ylabel, size_list, 
            func, plot_name, x_ticks_labels=False, max_val=2):
        '''Only for two differing scales'''
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot

        #ax.axvspan(0, 1, color='grey', alpha=0.5) # x values
        # grey out plot below a certain value
        #ax.axhspan(0, 1, color='grey', alpha=0.2, edgecolor='None') # y values

        print(x_list)
        print(y_list)

        lines = []
        lines2 = []
        count = 0
        for x, y, err, size, label, c, ls, m in zip(x_list, y_list, 
                error_list, size_list, 
                label_list, color_list, ls_list, marker_list):
            print('***', len(x), len(y))
            print('---', x, y, err)
            print()

            line = ax.scatter(x, y, label=label, 
                    s=size, facecolors=c, #'none', #c, #
                    edgecolors=c, marker=m, lw=2,
                    )
            ax.errorbar(x, y, yerr=err, c=c, ls=ls, ecolor=c, capsize=2,
                    lw=2)
            #line2 = ax.plot(x, y, c=c, lw=2, ls=ls,
                    #label=label
                    #)

            #lines.append(line)
            #lines2.append(line2)
            count += 1

        ax.set_ylim([-1, max_val])

        Plotter.format(ax, x, y, xlabel, ylabel)
        ax.grid(color='grey', linestyle='--', which='major', 
                axis='x', #'both'
                linewidth=1)
        #ax.yaxis.label.set_color(color_list[0])

        if x_ticks_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(x_ticks_labels, rotation='vertical'
                    )
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        labs = [l.get_label() for l in lines]
        lgd = ax.legend(lines, labs, bbox_to_anchor=(1.1, 1), 
                loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())



    def twinx_error_bars_plot(x_list, y_list, error_list, label_list, 
            color_list, ls_list, 
            marker_list, xlabel, ylabel, ylabel2, size_list, axis_list, 
            func, plot_name, x_ticks_labels=False, max_val=2):
        '''Only for two differing scales'''
        fig, ax = plt.subplots(figsize=(10, 10), 
                edgecolor='k') #all in one plot
        ax2 = ax.twinx()
        #ax.axvspan(0, 1, color='grey', alpha=0.5) # x values
        ax.axhspan(0, 1, color='grey', alpha=0.2, edgecolor='None') # y values
        print(x_list)
        print(y_list)

        lines = []
        lines2 = []
        count = 0
        for x, y, err, size, a, label, c, ls, m in zip(x_list, y_list, 
                error_list, size_list, 
                axis_list, label_list, color_list, ls_list, marker_list):
            #inds2 = x.argsort()
            #x = x[inds2]
            #y = y[inds2]
            print('***', len(x), len(y))
            print('---', x, y, err)
            print()
            if a == 1:
                line = ax.scatter(x, y, label=label, 
                        s=size, facecolors=c, #'none', #c, #
                        edgecolors=c, marker=m, lw=2,
                        )
                ax.errorbar(x, y, yerr=err, c=c, ls=ls, ecolor=c, capsize=2,
                        lw=2)
                #line2 = ax.plot(x, y, c=c, lw=2, ls=ls,
                        #label=label
                        #)
            if a == 2:
                line = ax2.scatter(x, y, label=label, 
                        s=size, facecolors=c, #'none', #
                        edgecolors=c, marker=m, lw=3,
                        )
                ax2.errorbar(x, y, yerr=err, c=c, ls=ls, ecolor=c, capsize=2,
                        lw=2)
                #line2 = ax2.plot(x, y, c=c, lw=2, ls=ls,
                        #label=label
                        #)





            '''
            x2, inds = np.unique(x, return_index=True)
            x_unique = x[inds]
            y_unique = y[inds]
            xnew = np.linspace(x_unique.min(), x_unique.max(), 1000)
            f_smooth = interp1d(x_unique, y_unique, kind=2)
            poly = np.polyfit(x_unique,y_unique,5)
            poly_y = np.poly1d(poly)(xnew)
            if func == 'f':
                curve = f_smooth(xnew)
                #poly = np.polyfit(x_unique,y_unique,27)
                #poly_y = np.poly1d(poly)(xnew)
                #curve = poly_y
            if func  == 'p':
                curve = poly_y
            if a == 1:
                line2 = ax.plot(xnew, 
                        #f_smooth(xnew), 
                        curve, 
                        c=c, lw=2, ls=ls,
                        #label=label
                        )
            if a == 2:
                line2 = ax2.plot(xnew, 
                        #f_smooth(xnew), 
                        curve, 
                        c=c, lw=3, ls=ls,
                        #label=label
                        )
            '''
            #lines.append(line)
            #lines2.append(line2)
            count += 1


        #ax2.set_ylabel(ylabel2, fontsize=Plotter.axis_labels, 
                #weight='medium')

        ax.set_ylim([0, max_val])
        ax2.set_ylim([0, 105])

        Plotter.format(ax2, x, y, xlabel, ylabel2)
        Plotter.format(ax, x, y, xlabel, ylabel)
        #ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax2.yaxis.offsetText.set_fontsize(Plotter.tick_labels)
        ax.grid(color='grey', linestyle='--', which='major', 
                axis='x', #'both'
                linewidth=1)
        ax.yaxis.label.set_color(color_list[0])
        ax2.yaxis.label.set_color(color_list[1])

        if x_ticks_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(x_ticks_labels, rotation='vertical'
                    )
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        labs = [l.get_label() for l in lines]
        lgd = ax.legend(lines, labs, bbox_to_anchor=(1.1, 1), 
                loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())




    def xy_scatter(x_list, y_list, label_list, color_list, 
            xlabel, ylabel, size_list, plot_name, log=False):
        fig, ax = plt.subplots(
                figsize=(10, 10), 
                #figsize=(20, 10),
                edgecolor='k') #all in one plot
        lines = []
        for x, y, size, label, c in zip(x_list, y_list, size_list, 
                label_list, color_list):
            line = ax.scatter(x, y, label=label, 
                    s=size, facecolors=c, #'none', 
                    edgecolors=c, #lw=2,
                    )
            # line2 = ax.plot(x, y, lw=3, linestyle='dashed')
            lines.append(line)

            # plot a fitted line through points
            func = False # "f" # 
            if func:
                ls = "dashed"
                x2, inds = np.unique(x, return_index=True)
                x_unique = x[inds].squeeze()
                y_unique = y[inds].squeeze()
                print(x_unique.shape, y_unique.shape)
                xnew = np.linspace(x_unique.min(), x_unique.max(), 1000)
                f_smooth = interp1d(x_unique, y_unique, kind=2)
                poly = np.polyfit(x_unique,y_unique,5)
                poly_y = np.poly1d(poly)(xnew)
                if func == 'f':
                    curve = f_smooth(xnew)
                    #poly = np.polyfit(x_unique,y_unique,27)
                    #poly_y = np.poly1d(poly)(xnew)
                    #curve = poly_y
                if func  == 'p':
                    curve = poly_y
                line2 = ax.plot(xnew, 
                        #f_smooth(xnew), 
                        curve, 
                        c=c, lw=2, ls=ls,
                        #label=label
                        )



        Plotter.format(ax, x, y, xlabel, ylabel)
        ax.grid(color='grey', linestyle='--', which='major', 
                axis='x', linewidth=1)
        if log:
            ax.set_xscale('log')
        #lgd = Plotter.get_legend(ax, lines, label_list)
        '''
        lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                prop={'size': Plotter.tick_labels+4})
        lgd.get_frame().set_alpha(0)
        '''
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())



    def xy_scatter_density(x, y, xlabel, ylabel, plot_name):
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('jet')
        sc = ax.scatter(x, y, c=z, s=50, cmap=cm)
        Plotter.format(ax, x, y, xlabel, ylabel)
        cb_ax = fig.add_axes([0.83, 0.12, 0.05, 0.75])
        Plotter.colorbar(fig, cb_ax, sc, 'Density')
        fig.savefig('%s' % (plot_name), 
                #transparent=True,
                #bbox_extra_artists=(lgd,),
                bbox_inches='tight',
                )
        plt.close(plt.gcf())

    def plot_violin(x_list, x_list2, label_list, color_list, xlabel, ylabel, 
                plot_name):
        """
        x_list is a list of lists, for x values for each molecule

        https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
        """
        vert = False
        if vert:
            fig, ax = plt.subplots(figsize=(20, 10), 
                    edgecolor='k') #all in one plot
        else:
            fig, ax = plt.subplots(
                    figsize=(10, 20), # all molecules
                    #figsize=(20, 10), 
                    edgecolor='k') #all in one plot
            x_list = x_list[::-1]          
            x_list2 = x_list2[::-1]          
            label_list = label_list[::-1]          

        sc1 = ax.violinplot(x_list, #list of vals
                            vert=vert,
                            widths=0.95, #width of each violin dist
                            showextrema=False, #remove min/max line
                        )
        
        for pc in sc1["bodies"]:
            # plt.colors.to_rgba(c, alpha=None)
            #pc.set_facecolor(color_list[0])
            pc.set_facecolor("none")
            pc.set_edgecolor(color_list[0])
            pc.set_linestyle("dashed")
            pc.set_alpha(1) #0.8)
            pc.set_linewidth(3) 
            if vert:
                # for vertical plots
                paths = pc.get_paths()[0]
                mean = np.mean(paths.vertices[:, 0])
                # paths.vertices[:, 0][paths.vertices[:, 0] <= mean] = mean
                # to mirror violin plots
                #pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, mean)
            else:
                # for horizontal plots
                paths = pc.get_paths()[0]
                mean = np.mean(paths.vertices[:, 1])
                # paths.vertices[:, 1][paths.vertices[:, 1] <= mean] = mean
                # to mirror violin plots
                #pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], mean, np.inf)              


        sc2 = ax.violinplot(x_list2, #list of vals
                            vert=vert,
                            widths=0.95, #width of each violin dist
                            showextrema=False, #remove min/max line
                        )
        for pc in sc2["bodies"]:
            #pc.set_facecolor(color_list[1])
            pc.set_facecolor("none")
            #pc.set_edgecolor("tab:blue")
            pc.set_edgecolor(color_list[1])
            pc.set_linestyle("dashed")
            pc.set_alpha(1) #0.8)
            pc.set_linewidth(3) 
            if vert:
                # for vertical plots
                paths = pc.get_paths()[0]
                mean = np.mean(paths.vertices[:, 0])
                # paths.vertices[:, 0][paths.vertices[:, 0] <= mean] = mean
                # to mirror violin plots
                #pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], mean, np.inf)
            else:
                # for horizontal plots
                paths = pc.get_paths()[0]
                mean = np.mean(paths.vertices[:, 1])
                #paths.vertices[:, 1][paths.vertices[:, 1] <= mean] = mean
                # to mirror violin plots
                #pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], -np.inf, mean)                


        Plotter.format(ax, x_list[0], x_list2[0], ylabel, xlabel)


        if vert:
            ax.set_xticks(np.arange(1, len(label_list) + 1), labels=label_list, 
                        rotation='vertical'
                        )
            ax.set_xlim(0.25, len(label_list) + 0.75)
            # ax.set_xlabel('Sample name')
            ax.grid(color='grey', linestyle='--', which='major', 
                axis='y', linewidth=1)
        else:
            ax.set_yticks(np.arange(1, len(label_list) + 1), labels=label_list, 
                        #rotation='vertical'
                        )
            ax.set_ylim(0.25, len(label_list) + 0.75)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

            ax.grid(color='grey', linestyle='--', which='major', 
                axis='x', linewidth=1)

            #ax.yaxis.tick_right()
            #ax.yaxis.set_label_position("right")
            # plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
            # ax.set_xlabel('Sample name')            



        fig.savefig('%s' % (plot_name), 
                #transparent=True, 
                bbox_inches='tight'
                )
        plt.close(plt.gcf())
