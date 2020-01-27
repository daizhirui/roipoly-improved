"""Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.
"""

import sys
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, Slider, TextBox
from collections import defaultdict

logger = logging.getLogger(__name__)

warnings.simplefilter('always', DeprecationWarning)


def deprecation(message):
    warnings.warn(message, DeprecationWarning)


class RoiPoly:

    def __init__(self, roi_val, fig=None, ax=None, color='b',
                 roicolor=None, show_fig=True, close_fig=True,
                 completed_callback=None):
        """

        Parameters
        ----------
        fig: matplotlib figure
            Figure on which to create the ROI
        ax: matplotlib axes
            Axes on which to draw the ROI
        color: str
           Color of the ROI
        roicolor: str
            deprecated, use `color` instead
        show_fig: bool
            Display the figure upon initializing a RoiPoly object
        close_fig: bool
            Close the figure after finishing ROI drawing
        """

        if roicolor is not None:
            deprecation("Use 'color' instead of 'roicolor'!")
            color = roicolor

        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()

        self.value = roi_val
        self.start_point = []
        self.end_point = []
        self.previous_point = []
        self.x = []
        self.y = []
        self.line = None
        self.completed = False  # Has ROI drawing completed?
        self.color = color
        self.fig = fig
        self.ax = ax
        self.close_figure = close_fig
        self.completed_callback = completed_callback

        # Mouse event callbacks
        self.__cid1 = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.__motion_notify_callback)
        self.__cid2 = self.fig.canvas.mpl_connect(
            'button_press_event', self.__button_press_callback)

        if show_fig:
            self.show_figure()

    @staticmethod
    def show_figure():
        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show(block=True)

    def get_mask(self, current_image_shape):
        ny, nx = current_image_shape
        poly_vertices = ([(self.x[0], self.y[0])]
                         + list(zip(reversed(self.x), reversed(self.y))))
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_vertices)
        grid = roi_path.contains_points(points).reshape((ny, nx))
        return grid

    def display_roi(self, **linekwargs):
        line = plt.Line2D(self.x + [self.x[0]], self.y + [self.y[0]],
                          color=self.color, **linekwargs)
        ax = plt.gca()
        ax.add_line(line)
        plt.draw()

    def get_mean_and_std(self, current_image):
        mask = self.get_mask(current_image)
        mean = np.mean(np.extract(mask, current_image))
        std = np.std(np.extract(mask, current_image))
        return mean, std

    def display_mean(self, current_image, **textkwargs):
        mean, std = self.get_mean_and_std(current_image)
        string = "%.3f +- %.3f" % (mean, std)
        plt.text(self.x[0], self.y[0],
                 string, color=self.color,
                 bbox=dict(facecolor='w', alpha=0.6), **textkwargs)

    def __motion_notify_callback(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if ((event.button is None or event.button == 1) and
                    self.line is not None):
                # Move line around
                x_data = [self.previous_point[0], x]
                y_data = [self.previous_point[1], y]
                logger.debug("draw line x: {} y: {}".format(x_data, y_data))
                self.line.set_data(x_data, y_data)
                self.fig.canvas.draw()

    def __button_press_callback(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1 and event.dblclick is False:
                logger.debug("Received single left mouse button click")
                if self.line is None:  # If there is no line, create a line
                    self.line = plt.Line2D([x, x], [y, y],
                                           marker='o', color=self.color)
                    self.start_point = [x, y]
                    self.previous_point = self.start_point
                    self.x = [x]
                    self.y = [y]

                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # Add a segment
                else:
                    # If there is a line, create a segment
                    x_data = [self.previous_point[0], x]
                    y_data = [self.previous_point[1], y]
                    logger.debug(
                        "draw line x: {} y: {}".format(x_data, y_data))
                    self.line = plt.Line2D(x_data, y_data,
                                           marker='o', color=self.color)
                    self.previous_point = [x, y]
                    self.x.append(x)
                    self.y.append(y)

                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()

            elif (((event.button == 1 and event.dblclick is True) or
                   (event.button == 3 and event.dblclick is False)) and
                  self.line is not None):
                # Close the loop and disconnect
                logger.debug("Received single right mouse button click or "
                             "double left click")
                self.fig.canvas.mpl_disconnect(self.__cid1)
                self.fig.canvas.mpl_disconnect(self.__cid2)

                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None
                self.completed = True

                if self.completed_callback is not None:
                    self.completed_callback(self)

                if not sys.flags.interactive and self.close_figure:
                    #  Figure has to be closed so that code can continue
                    plt.close(self.fig)

    # For compatibility with old version
    def displayMean(self, *args, **kwargs):
        deprecation("Use 'display_mean' instead of 'displayMean'!")
        return self.display_mean(*args, **kwargs)

    def getMask(self, *args, **kwargs):
        deprecation("Use 'get_mask()' instead of 'getMask'!")
        return self.get_mask(*args, **kwargs)

    def displayROI(self, *args, **kwargs):
        deprecation("Use 'display_roi' instead of 'displayROI'!")
        return self.display_roi(*args, **kwargs)


class MultiRoi:
    def __init__(self, img=None, roi_types=(),
                 color_cycle=('b', 'g', 'r', 'c', 'm', 'y', 'k'),
                 finish_callback=None):

        self.color_cycle = color_cycle
        self.mask_fig = plt.figure(figsize=(5, 3))
        self.mask_fig_idx = 1
        self.mask_axes = self.mask_fig.gca()
        self.mask_axes.set_title("Mask")
        self.mask_axes.set_axis_off()
        self.img_fig = plt.figure(figsize=(10, 6))
        self.img_fig_idx = 2
        self.img_axes = self.img_fig.gca()
        self.img_axes.set_axis_off()
        self.__image = None
        self.image = img
        # __mask is for visualization, mask is for labling
        self.__mask = None if img is None else np.zeros(img.shape)
        self.mask = None if img is None else np.zeros(img.shape[:2])
        self.rois = []
        self.roi_values = {}
        self.roi_value_colors = {}
        self.current_roi_value = None
        self.finish_callback = finish_callback
        self.textbox = None
        self.buttons = []
        for roi_name in roi_types:
            self.__add_roi_type(roi_name)
        self.make_widgets()
        self.update_plot()

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, img):
        self.__image = img
        self.__mask = None if img is None else np.zeros(img.shape)
        self.mask = None if img is None else np.zeros(img.shape[:2],
                                                      dtype=np.uint8)
        if self.__image is not None:
            self.update_plt_mask()
            self.update_plt_img()
        self.rois = []  # empty rois

    @staticmethod
    def update_plot():
        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show(block=True)
        # plt.show(block=True)

    def make_widgets(self):
        self.textbox = TextBox(ax=plt.axes([0.2, 0.07, 0.1, 0.04]),
                               label="ROI Name: ", initial='[roi_name]')
        btn_finish = Button(plt.axes([0.8, 0.07, 0.1, 0.04]), 'Finish')
        btn_finish.on_clicked(self.finish)
        btn_clear = Button(plt.axes([0.7, 0.07, 0.1, 0.04]), 'Clear')
        btn_clear.on_clicked(self.clear_rois)
        btn_add = Button(plt.axes([0.6, 0.07, 0.1, 0.04]), 'New ROI')
        btn_add.on_clicked(self.new_roi)
        btn_new = Button(plt.axes([0.5, 0.07, 0.1, 0.04]), 'New ROI Type')
        btn_new.on_clicked(self.new_roi_type)
        self.buttons.extend([btn_new, btn_add, btn_clear, btn_finish])
        # plt.show(block=True)

    def completed(self):
        if len(self.rois) > 0:
            return all(r.completed for r in self.rois)
        else:
            return True

    def __add_roi_type(self, roi_name):
        if roi_name in self.roi_values.values():
            return
        roi_value = len(self.roi_values.keys()) + 1
        self.roi_values[roi_value] = roi_name
        self.roi_value_colors[roi_value] = np.random.rand(3)
        btn = Button(plt.axes([0.1 * roi_value, 0.02, 0.1, 0.04]), roi_name)
        btn.on_clicked(lambda x: self.select_roi_type(roi_value, x))
        self.buttons.append(btn)
        return roi_value

    def select_roi_type(self, idx, event):
        self.textbox.set_val(self.roi_values[idx])
        self.current_roi_value = idx

    def new_roi_type(self, event):
        roi_name = self.textbox.text
        if roi_name in self.roi_values.values():
            msg = "{} ROI already exists.".format(roi_name)
            self.log(logging.WARNING, msg)
            return
        # roi_value begins from 1
        roi_value = self.__add_roi_type(roi_name)
        self.roi_values[roi_value] = roi_name
        # generate a color randomly for this type of ROI
        self.roi_value_colors[roi_value] = np.random.rand(3)
        if self.completed():
            self.current_roi_value = roi_value
        msg = "Add New ROI Type: {}".format(roi_name)
        self.log(logging.INFO, msg)
        self.update_plot()

    def new_roi(self, event):
        if self.__image is None:
            self.log(logging.WARNING, "No image loaded yet.")
            return
        if not self.completed():
            self.log(logging.INFO, "Some roi(s) haven't been completed.")
            return
        rois_cnt = len(self.rois)
        try:
            roi_name = self.roi_values[self.current_roi_value]
        except KeyError:
            return
        msg = "Creating new ROI {}, type {}".format(rois_cnt, roi_name)
        self.log(logging.INFO, msg)
        plt.draw()
        roi_color = self.color_cycle[rois_cnt % len(self.color_cycle)]
        roi = RoiPoly(roi_val=self.current_roi_value,
                      color=roi_color, fig=self.img_fig, ax=self.img_axes,
                      close_fig=False, show_fig=False,
                      completed_callback=self.roi_completed)
        self.rois.append(roi)

    def roi_completed(self, roi: RoiPoly):
        msg = "ROI {} completed".format(roi.value)
        self.log(logging.INFO, msg)
        mask = roi.get_mask(self.__image.shape[:2])
        self.mask[mask] = roi.value
        self.__mask[mask] = self.roi_value_colors[roi.value]
        self.update_plt_mask()

    def clear_rois(self, event):
        self.image = np.array(self.__image)
        self.update_plot()

    def finish(self, event):
        if not self.completed():
            self.log(logging.INFO, "Some roi(s) haven't been completed")
            return
        if self.finish_callback:
            if self.finish_callback(self):
                logging.log(logging.INFO, "Finishing signal from callback.")
                plt.close('all')
            else:
                pass
        else:
            self.log(logging.WARNING, "No callback after finishing a ROI.")
            plt.close('all')

    def update_img_title(self, title):
        plt.figure(self.img_fig_idx)
        self.img_axes.set_title(title)
        plt.draw()

    def update_mask_title(self, title):
        plt.figure(self.mask_fig_idx)
        plt.title(title)
        plt.draw()
        plt.figure(self.img_fig_idx)

    def update_plt_img(self):
        plt.figure(self.img_fig_idx)
        self.img_axes.clear()
        self.img_axes.set_axis_off()
        self.img_axes.imshow(self.__image)
        plt.draw()

    def update_plt_mask(self):
        plt.figure(self.mask_fig_idx)
        self.mask_axes.imshow(self.__mask)
        plt.draw()
        plt.figure(self.img_fig_idx)

    def log(self, level, msg):
        logging.log(level, msg)
        self.update_img_title(msg)


# For compatibility with old version
def roipoly(*args, **kwargs):
    deprecation("Import 'RoiPoly' instead of 'roipoly'!")
    return RoiPoly(*args, **kwargs)
