import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class SpectralPlotter:
    def __init__(self, show_plot="no"):
        """
        Initialize a SpectralPlotter object.

        Args:
            show_plot (str): A string indicating whether to display the plot. Should be 'yes' or 'no'.
        """
        self.show_plot = show_plot
        self.S1_4_WL = range(1100, 1352, 2)
        self.S1_7_WL = range(1350, 1652, 2)
        self.S2_0_WL = range(1550, 1952, 2)
        self.S2_2_WL = range(1750, 2152, 2)

    def plot_based_on_classes(self, all_WL, all_I, class_all, plot_label, title, xlabel, ylabel):
        """
        Plot data based on different classes.

        Args:
            all_WL (array-like): Array of wavelength values.
            all_I (array-like): Array of intensity values.
            class_all (array-like): Array of class labels for each data point.
            plot_label (list): List of labels for the plot legend.
            title (str): Title for the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.

        Returns:
            fig (matplotlib.figure.Figure or None): The generated figure object if show_plot is 'yes', otherwise None.
        """
        mpl.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.linewidth'] = 1

        font_size_title = 6 + 2
        font_size_ylabel = 6 + 2
        font_size_xlabel = 6 + 2
        font_size_ticks = 4 + 2
        font_size_legend = 4 + 2
        grid_line_width = 0.5
        plot_line_width = 0.5
        plot_marker = 'None'
        plot_ls_0 = 'solid'
        plot_ls_1 = 'dashed'
        plot_marker_sz = 1.0
        alpha_value = 0.5

        pop_a = mpatches.Patch(color='green', label=plot_label[0])
        pop_b = mpatches.Patch(color='magenta', label=plot_label[1])

        if self.show_plot == "yes":
            fig = plt.figure(figsize=(7.00, 3.50), dpi=300)

            for i in range(np.shape(all_I)[0]):
                if class_all[i] == 0:
                    plt.plot(
                        all_WL,
                        all_I[i],
                        color='green',
                        linewidth=plot_line_width,
                        marker=plot_marker,
                        linestyle=plot_ls_0,
                        markersize=plot_marker_sz,
                        alpha=alpha_value,
                    )
                elif class_all[i] == 1:
                    plt.plot(
                        all_WL,
                        all_I[i],
                        color='magenta',
                        linewidth=plot_line_width,
                        marker=plot_marker,
                        linestyle=plot_ls_1,
                        markersize=plot_marker_sz,
                        alpha=alpha_value,
                    )

            plt.ylabel(ylabel, fontsize=font_size_ylabel)
            plt.xlabel(xlabel, fontsize=font_size_xlabel)
            plt.title(title, fontsize=font_size_title)

            plt.legend(handles=[pop_a, pop_b], prop={'size': font_size_legend}, loc='upper right')

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

            plt.gca().xaxis.set_tick_params(which='major', direction='in', top=False)
            plt.gca().yaxis.set_tick_params(which='major', direction='in', right=False)
            plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)

            plt.gca().yaxis.grid(True, linewidth=grid_line_width, ls='dotted')

            plt.xticks([])

            end_S1_4_WL = len(self.S1_4_WL)
            end_S1_7_WL = len(self.S1_4_WL) + len(self.S1_7_WL)
            end_S2_0_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL)
            end_S2_2_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL) + len(self.S2_2_WL)

            x_ticks_pos = [
                0,
                int(len(self.S1_4_WL) / 2),
                end_S1_4_WL - 1,
                end_S1_4_WL,
                end_S1_4_WL + int(len(self.S1_7_WL) / 2),
                end_S1_7_WL - 1,
                end_S1_7_WL,
                end_S1_7_WL + int(len(self.S2_0_WL) / 2),
                end_S2_0_WL - 1,
                end_S2_0_WL,
                end_S2_0_WL + int(len(self.S2_2_WL) / 2),
                end_S2_2_WL - 1,
            ]

            x_ticks_label = [
                str(self.S1_4_WL[0]),
                "...",
                str(self.S1_4_WL[-1]),
                str(self.S1_7_WL[0]),
                "...",
                str(self.S1_7_WL[-1]),
                str(self.S2_0_WL[0]),
                "...",
                str(self.S2_0_WL[-1]),
                str(self.S2_2_WL[0]),
                "...",
                str(self.S2_2_WL[-1]),
            ]

            plt.xticks(x_ticks_pos, x_ticks_label)
            ticklabels = plt.gca().get_xticklabels()

            ticklabels[0].set_ha("left")
            ticklabels[2].set_ha("right")
            ticklabels[3].set_ha("left")
            ticklabels[5].set_ha("right")
            ticklabels[6].set_ha("left")
            ticklabels[8].set_ha("right")
            ticklabels[9].set_ha("left")
            ticklabels[-1].set_ha("right")

            y_min, y_max = plt.gca().get_ylim()
            plt.vlines(x=end_S1_4_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S1_7_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S2_0_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)

            plt.axvspan(end_S1_4_WL, end_S1_7_WL - 1, alpha=0.1)
            plt.axvspan(end_S2_0_WL, end_S2_2_WL - 1, alpha=0.1)

            fig.set_tight_layout(True)
            plt.show()
        else:
            fig = None
        return fig
