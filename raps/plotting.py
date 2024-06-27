"""
This module provides functionality for creating and saving various types of plots using Matplotlib.

The module defines a `BasePlotter` class for setting up plots and saving them, and a `Plotter` class
that extends `BasePlotter` to include methods for plotting histories, histograms, and comparisons.

Classes
-------
BasePlotter
    A base class for setting up and saving plots.
Plotter
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.
"""

import matplotlib.pyplot as plt
from uncertainties import unumpy

class BasePlotter:
    """
    A base class for setting up and saving plots.

    Attributes
    ----------
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The title of the plot.
    """
    def __init__(self, xlabel, ylabel, title, uncertainties=False):
        """
        Constructs all the necessary attributes for the BasePlotter object.

        Parameters
        ----------
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        title : str
            The title of the plot.
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.uncertainties = uncertainties

    def setup_plot(self, figsize=(10, 5)):
        """
        Sets up the plot with the given figure size, labels, title, and grid.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure (default is (10, 5)).
        """
        plt.figure(figsize=figsize)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)

    def save_and_close_plot(self, save_path):
        """
        Saves the plot to the specified path and closes the plot.

        Parameters
        ----------
        save_path : str
            The path to save the plot.
        """
        plt.savefig(save_path)
        plt.close()

class Plotter(BasePlotter):
    """
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.

    Attributes
    ----------
    save_path : str
        The path to save the plot.
    """
    def __init__(self, xlabel='', ylabel='', title='', save_path='out.svg', uncertainties=False):
        """
        Constructs all the necessary attributes for the Plotter object.

        Parameters
        ----------
        xlabel : str, optional
            The label for the x-axis (default is an empty string).
        ylabel : str, optional
            The label for the y-axis (default is an empty string).
        title : str, optional
            The title of the plot (default is an empty string).
        save_path : str, optional
            The path to save the plot (default is 'out.svg').
        uncertainties: boolean, optional
            Flag if uncertainties are enabled and ufloats are used.
        """
        super().__init__(xlabel, ylabel, title, uncertainties)
        self.save_path = save_path

    def plot_history(self, x, y):
        """
        Plots a history plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()

        if self.uncertainties:
            nominal_curve = plt.plot(x, unumpy.nominal_values(y))
            plt.fill_between(x, unumpy.nominal_values(y)-unumpy.std_devs(y),
                             unumpy.nominal_values(y)+unumpy.std_devs(y),
                             facecolor=nominal_curve[0].get_color(),
                             edgecolor='face', alpha=0.1, linewidth=0)
        else:
            plt.plot(x, y)
        self.save_and_close_plot(self.save_path)

    def plot_histogram(self, data, bins=50):
        """
        Plots a histogram of the given data and saves it.

        Parameters
        ----------
        data : list
            The data to plot in the histogram.
        bins : int, optional
            The number of bins in the histogram (default is 50).
        """
        self.setup_plot()
        plt.hist(data, bins=bins)
        self.save_and_close_plot(self.save_path)

    def plot_compare(self, x, y):
        """
        Plots a comparison plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()
        plt.plot(x, y)
        self.save_and_close_plot(self.save_path)


if __name__ == "__main__":
    plotter = Plotter()
    #plotter.plot_history([1, 2, 3, 4])
