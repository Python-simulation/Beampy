"""
The user_interface module is the main file of the beampy module.
It contain the UserInterface class used to computed and displayed the bpm
results onto the interface.

This module was at first developed by Marcel Soubkovsky for the implementation
of the array of guides, of one gaussian beam and of the plotting methods.
Then, continued by Jonathan Peltier.
"""
import sys
import os
import webbrowser
import time
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog)
from PyQt5.QtCore import (pyqtSlot, Qt, QSize)
from PyQt5.QtGui import QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib import lines
from matplotlib.patches import Polygon

# interface.py generated from interface.ui made using Qt designer
from beampy.interface import Ui_MainWindow
from beampy.bpm import Bpm  # Bpm class with all the BPM methods

# ! To translate from .ui to .py -> pyuic5 -x interface.ui -o interface.py

# Check out if doubts on the interface:
# http://blog.rcnelson.com/building-a-matplotlib-gui-with-qt-designer-part-1/


class UserInterface(QMainWindow, Ui_MainWindow):
    """
    This class connect the :class:`.Bpm` class from the :mod:`.bpm` module
    with the :meth:`.setupUi` method from the :mod:`.interface` module.
    """

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)  # Linked to Ui_MainWindow through interface.py

        # Prepare variables before assigning values to them
        self.width = []
        self.offset_guide = []
        self.delta_no = []
        self.shape_gauss_check = []
        self.gauss_pow = np.array([], dtype=int)
        self.shape_squared_check = []
        self.nbr_p = np.array([], dtype=int)
        self.p = []
        self.curve = []
        self.half_delay = []
        self.distance_factor = []
        self.tab_index = []
        self.previous_guide = 0

        self.fwhm = []
        self.offset_light = []
        self.irrad = []
        self.offset_check = []
        self.gaussian_check = []
        self.square_check = []
        self.mode_check = []
        self.all_modes_check = []
        self.mode = np.array([], dtype=int)
        self.mode_guide_ref = np.array([], dtype=int)
        self.offset_light_peak = np.array([], dtype=int)
        self.airy_check = []
        self.airy_zero = np.array([], dtype=int)
        self.lobe_size = []
        self.previous_beam = 0

        self.on_click_create_guide()  # Initialized variables with buttons
        self.on_click_create_light()  # Initialized variables with buttons

        self.calculate_guide()  # Compute guide
        self.calculate_light()  # Compute light

        self.addmpl('guide')  # Display guide
        self.addmpl('light')  # Display light

        self.show_estimate_time()

        # Initialized compute graphics
        self.canvas_compute_propag = FigureCanvas(Figure())
        self.plot_compute.addWidget(self.canvas_compute_propag)
        self.toolbar_3 = FigureCanvas(Figure())
        self.plot_compute.addWidget(self.toolbar_3)
        self.toolbar_3.close()
        self.canvas_5 = FigureCanvas(Figure())
        self.verticalLayout_compute_plot.addWidget(self.canvas_5)
        self.toolbar_5 = FigureCanvas(Figure())
        self.verticalLayout_compute_plot.addWidget(self.toolbar_5)
        self.canvas_compute_end = FigureCanvas(Figure())
        self.plot_compute.addWidget(self.canvas_compute_end)
        self.toolbar_4 = FigureCanvas(Figure())
        self.plot_compute.addWidget(self.toolbar_4)

        self.filename = None

        self.connect_buttons()
        self.create_menu()

    def connect_buttons(self):
        """
        Connect the interface buttons to their corresponding functions:

        :meth:`.on_click_guide`, :meth:`.on_click_curved`,
        :meth:`.on_click_light`, :meth:`.on_click_compute`,
        :meth:`.on_click_create_light`, :meth:`.on_click_delete_light`,
        :meth:`.save_light`, :meth:`.get_guide`,
        :meth:`.get_light`, :meth:`.get_compute`,
        :meth:`.show_estimate_time`, :meth:`.check_modes_display`.
        """
        self.calculateButton_guide.clicked.connect(self.on_click_guide)
        self.calculateButton_light.clicked.connect(self.on_click_light)
        self.calculateButton_compute.clicked.connect(self.on_click_compute)
        self.pushButton_create_beam.clicked.connect(self.on_click_create_light)
        self.pushButton_delete_beam.clicked.connect(self.on_click_delete_light)
        self.pushButton_save_beam.clicked.connect(self.save_light)
        self.pushButton_create_guide.clicked.connect(
            self.on_click_create_guide)
        self.pushButton_delete_guide.clicked.connect(
            self.on_click_delete_guide)
        self.pushButton_save_guide.clicked.connect(self.save_guide)
        self.pushButton_cancel_guide.clicked.connect(self.get_guide)
        self.pushButton_cancel_light.clicked.connect(self.get_light)
        self.pushButton_cancel_compute.clicked.connect(self.get_compute)
        self.pushButton_estimate_time.clicked.connect(self.show_estimate_time)
        self.pushButton_mode_number.clicked.connect(self.check_modes_display)

    def create_menu(self):
        """
        Create a menu to open, save a file, or exit the app.

        Notes
        -----
        This method connect the following methods and function to the
        menu buttons:

        :meth:`.open_file_name`, :meth:`.save_quick`, :meth:`.save_file_name`,
        :func:`.open_doc`.
        """
        folder = __file__  # Module name
        # Replaces characters only when called from outer files
        folder = folder.replace("\\", "/")
        folder = folder.split("/")
        folder = folder[:-1]  # Remove the file name
        folder2 = str()

        for line in folder:
            folder2 = folder2+"/"+line

        folder = folder2[1:]+"/"

        icon = QIcon()
        icon.addFile(folder+'icons/beampy-logo.png', QSize(256, 256))
        self.setWindowIcon(icon)

        menubar = self.menuBar()

        file = menubar.addMenu('File')

        action = file.addAction('Open')
        action.triggered.connect(self.open_file_name)
        action.setShortcut('Ctrl+O')
        icon = QIcon()
        icon.addFile(folder+'icons/document-open.png', QSize(22, 22))
        action.setIcon(icon)

        action = file.addAction('Save')
        action.triggered.connect(self.save_quick)
        action.setShortcut('Ctrl+S')
        icon = QIcon()
        icon.addFile(folder+'icons/document-save.png', QSize(22, 22))
        action.setIcon(icon)

        action = file.addAction('Save as')
        action.triggered.connect(self.save_file_name)
        action.setShortcut('Ctrl+Shift+S')
        icon = QIcon()
        icon.addFile(folder+'icons/document-save-as.png', QSize(22, 22))
        action.setIcon(icon)

        action = file.addAction('Exit')  # Clean exit for spyder
        action.setShortcut('Ctrl+Q')
        action.triggered.connect(QApplication.quit)
        icon = QIcon()
        icon.addFile(folder+'icons/application-exit.png', QSize(22, 22))
        action.setIcon(icon)

        file = menubar.addMenu('Help')

        action = file.addAction('Documentation')
        action.triggered.connect(open_doc)
        icon = QIcon()
        icon.addFile(folder+'icons/help-about.png', QSize(22, 22))
        action.setIcon(icon)

    def calculate_guide(self):
        """
        Initialized the :class:`.Bpm` class and creates the guides.

        Notes
        -----
        Creats many variables, including :

        - peaks : Central position of each guide [guide,z].
        - dn : Difference of reefractive index [z,x].

        This method calls the following methods from :class:`.Bpm`:

        :meth:`.create_x_z`, :meth:`.squared_guide`, :meth:`.gauss_guide`,
        :meth:`.create_guides`, :meth:`.create_curved_guides`.
        """
        print("calculate guide")
        t_guide_start = time.process_time()

        self.save_guide()  # Get all guide values

        # Create the Bpm class (overwrite existing one)
        self.bpm = Bpm(self.no,
                       self.length_z, self.dist_z, self.nbr_z_disp,
                       self.length_x, self.dist_x)

        # windows variables
        [self.length_z, self.nbr_z, self.nbr_z_disp,
         self.length_x, self.nbr_x, self.x] = self.bpm.create_x_z()

        self.doubleSpinBox_length_x.setValue(self.length_x)
        self.doubleSpinBox_length_z.setValue(self.length_z)
        self.spinBox_nbr_z_disp.setValue(self.nbr_z_disp)

        if (self.nbr_x * self.nbr_z_disp) > (5000 * 1000):
            print("Error: if you want to have more points,")
            print("change the condition nbr_x * nbr_z_disp in calculate_guide")
            raise RuntimeError("Too many points:", self.nbr_x*self.nbr_z_disp)

        if (self.nbr_x * self.nbr_z) > (10000 * 15000) or (
                self.nbr_z > 40000 or self.nbr_x > 40000):
            print("Error: if you want to have more points,")
            print("change the condition nbr_x * nbr_z in calculate_guide")
            raise RuntimeError("Too many points:", self.nbr_x*self.nbr_z)

        nbr_guide = self.comboBox_guide.count()  # Number of guide set

        self.peaks = np.zeros((0, self.nbr_z))
        self.dn = np.zeros((self.nbr_z, self.nbr_x))

        for i in range(nbr_guide):
            # Waveguide shape choice
            if self.shape_squared_check[i]:  # Squared guides
                shape = self.bpm.squared_guide(self.width[i])
            elif self.shape_gauss_check[i]:  # Gaussian guides
                shape = self.bpm.gauss_guide(self.width[i],
                                             gauss_pow=self.gauss_pow[i])

            # Topology waveguides choice
            if self.tab_index[i] == 0:  # array of waveguides
                length_max = self.length_x - self.dist_x

                if self.nbr_p[i]*self.p[i] > length_max:
                    # give info about possible good values
                    print("nbr_p*p> length_x: ")
                    print(self.nbr_p[i]*self.p[i], ">", length_max)

                    print("p_max=", round(length_max/self.nbr_p[i], 3))

                    if int(length_max / self.p[i]) == length_max / self.p[i]:
                        print("nbr_p_max=", int(length_max / self.p[i])-1)
                    else:
                        print("nbr_p_max=", int(length_max / self.p[i]))

                    self.nbr_p[i] = 0
                [peaks, dn] = self.bpm.create_guides(
                    shape, self.delta_no[i], self.nbr_p[i], self.p[i],
                    offset_guide=self.offset_guide[i])

            elif self.tab_index[i] == 1:  # curved waveguides
                curve = self.curve[i] * 1E-8  # curvature factor
                [peaks, dn] = self.bpm.create_curved_guides(
                    shape, self.width[i], self.delta_no[i],
                    curve, self.half_delay[i], self.distance_factor[i],
                    offset_guide=self.offset_guide[i])

            if self.delta_no[i] > self.no/10:
                print("Careful: index variation is too high for guide", i, ":")
                print(self.delta_no[i], ">", self.no, "/ 10")

            self.peaks = np.append(self.peaks, peaks, 0)
            self.dn = np.add(self.dn, dn)

            # Display guides
            self.z_disp = np.linspace(0,
                                      self.length_z/1000,
                                      self.nbr_z_disp+1)

            self.xv, self.zv = np.meshgrid(self.x, self.z_disp)
            self.dn_disp = np.linspace(0,
                                       self.nbr_z-1,
                                       self.nbr_z_disp+1, dtype=int)

        # only display available settings
        # (don't propose offset from guide if no guide)
        if 0 in self.nbr_p or 0 in self.p:
            self.spinBox_offset_light_peak.setDisabled(True)
            self.spinBox_guide_nbr_ref.setDisabled(True)
            self.doubleSpinBox_offset_light.setEnabled(True)
            self.checkBox_offset_light.setChecked(False)
            self.checkBox_offset_light.setDisabled(True)
            self.offset_check *= 0
            self.checkBox_lost.setDisabled(True)
            self.checkBox_lost.setChecked(False)
            self.frame_lost.setDisabled(True)
            self.checkBox_check_power.setDisabled(True)
        else:
            self.checkBox_offset_light.setEnabled(True)
            self.spinBox_offset_light_peak.setMaximum(self.peaks.shape[0]-1)
            self.spinBox_guide_nbr_ref.setMaximum(self.peaks.shape[0]-1)
            self.checkBox_lost.setEnabled(True)
            self.spinBox_guide_lost.setMaximum(self.peaks.shape[0]-1)
            self.checkBox_check_power.setEnabled(True)

#        Define new min/max for light and looses, based on selected guides
        self.doubleSpinBox_width_lost.setMaximum(self.length_x-self.dist_x)
        self.doubleSpinBox_lobe_size.setMaximum(self.length_x-self.dist_x)
        # If want to use min/max for offset: multiple beams will have issue if
        # the windows size change (min/max will only be for the displayed beam)
#        self.doubleSpinBox_offset_light.setMinimum(self.x[0])
#        self.doubleSpinBox_offset_light.setMaximum(self.x[-1])

        self.calculate_guide_done = True
        t_guide_end = time.process_time()
        print('Guides time: ', t_guide_end-t_guide_start)

    def calculate_light(self):
        """
        Create the choosen beams.

        Notes
        -----
        Creates the progress_pow variable.

        This method calls the following methods from the :class:`.Bpm` class:

        :meth:`.gauss_light`, :meth:`.squared_light`, :meth:`.all_modes`,
        :meth:`.mode_light`, :meth:`.airy_light`, :meth:`.init_field`.
        """
        print("calculate light")
        t_light_start = time.process_time()

        self.save_light()  # Get all light variables
        # must have same wavelength and angle or must compute for each
        # different wavelength or angle
        nbr_light = self.comboBox_light.count()  # Number of beams
        field = np.zeros((nbr_light, self.nbr_x))

        if 1 in self.all_modes_check:  # Display only once the max mode
            self.check_modes_display()

        for i in range(nbr_light):

            # Check if offset relative to guide number or else in µm
            if self.offset_check[i] and self.peaks.shape[0] != 0:

                # Reduce light # if > guides #
                peaks_i = self.offset_light_peak[i]
                peaks_max = self.spinBox_offset_light_peak.maximum()

                if peaks_i > peaks_max:
                    print("beam", i+1, "has a non-existing guide position")
                    print("Change position from", peaks_i, "to", peaks_max)
                    self.offset_light_peak[i] = peaks_max

                offset_light = self.peaks[self.offset_light_peak[i], 0]

            else:
                offset_light = self.offset_light[i]

            guide_index = self.find_guide_number(self.mode_guide_ref[i])

            #  Compute lights
            if self.gaussian_check[i]:
                field_i = self.bpm.gauss_light(
                    self.fwhm[i], offset_light=offset_light)

            elif self.square_check[i]:
                field_i = self.bpm.squared_light(
                    self.fwhm[i], offset_light=offset_light)

            elif self.all_modes_check[i]:
                field_i = self.bpm.all_modes(
                    self.width[guide_index], self.delta_no[guide_index],
                    self.lo, offset_light=offset_light)[0]

            elif self.mode_check[i]:

                try:
                    field_i = self.bpm.mode_light(
                        self.width[guide_index], self.delta_no[guide_index],
                        self.mode[i], self.lo, offset_light=offset_light)[0]

                except ValueError as ex:  # Say that no mode exist
                    print(ex, "for the beam", i+1)
                    continue  # Go to the next field

            elif self.airy_check[i]:
                [field_i, last_zero_pos] = self.bpm.airy_light(
                    self.lobe_size[i], self.airy_zero[i],
                    offset_light=offset_light)
                self.spinBox_airy_zero.setValue(last_zero_pos)  # Corrected val

            field[i] = field_i

        irrad = self.irrad  # Irradiance or power (GW/cm^2)
        irrad = irrad * 1E13  # (W/m^2)
        [self.progress_pow] = self.bpm.init_field(
            self.dn, field, self.theta_ext, irrad, self.lo)

        t_light_end = time.process_time()
        print('light time: ', t_light_end-t_light_start)

    def calculate_propagation(self):
        """
        Calculate the propagation based on the input light and guides shapes.

        Notes
        -----
        Creates the progress_pow variable.

        Calls the following methods from :class:`.Bpm`:
        :meth:`.losses_position`, meth`.main_compute`.
        """
        print("calculate propagation")
        t_compute_start = time.process_time()
        self.calculate_guide_done = False

        self.save_compute()

        kerr = self.kerr_check  # Active Kerr effect
        kerr_loop = self.kerr_loop
        chi3 = self.chi3 * 1E-20  # m^2/V^2
        variance_check = self.variance_check

        if self.lost_check:
            alpha = self.alpha/1000
            [lost_beg, lost_end] = self.bpm.losses_position(
                self.peaks, self.guide_lost, self.width_lost)
        else:
            alpha = 0
            lost_beg = 0
            lost_end = 0

        estimation = round(
            8.8 / 5e7 * self.nbr_z * self.nbr_x  # without kerr
            * (1 + 0.72*self.kerr_check*(self.kerr_loop))  # with kerr
            + 3.8/5e6*self.nbr_z*self.nbr_x*self.variance_check  # control
            + 9/1e7*self.nbr_z*self.width_lost[0]/self.dist_x*self.lost_check,
            1)  # looses
        print("Time estimate:", estimation)

        [self.progress_pow] = self.bpm.main_compute(
            self.dn,
            chi3=chi3, kerr=kerr, kerr_loop=kerr_loop,
            variance_check=variance_check, alpha=alpha,
            lost_beg=lost_beg, lost_end=lost_end)

        t_compute_end = time.process_time()
        print('Compute time: ', t_compute_end-t_compute_start)

    def show_estimate_time(self):
        """
        Display - on the interface - the estimate time needed to compute the
        propagation, based on linearized experimental values.
        The estimation takes into account the looses, Kerr, and control
        parameters.
        """
        self.save_compute()
        estimation = round(
            8.8 / 5e7 * self.nbr_z * self.nbr_x  # without kerr
            * (1 + 0.72*self.kerr_check*(self.kerr_loop))  # with kerr
            + 3.8/5e6*self.nbr_z*self.nbr_x*self.variance_check  # control
            + 9/1e7*self.nbr_z*self.width_lost[0]/self.dist_x*self.lost_check,
            1)  # looses

        self.estimate_time_display.display(estimation)

    def find_guide_number(self, guide_number):
        """
        Return the waveguide group number for a given waveguide number.


        Parameters
        ----------
        guide_number : int
            Number of a waveguide

        Returns
        -------
        guide_group_number : int
            Number of the waveguide group
        """
        num_gd = -1
        for j, n in enumerate(self.nbr_p):

            if self.tab_index[j] == 0:

                for _ in range(n):
                    num_gd += 1
                    if num_gd == guide_number:
                        guide_group_number = j
                        break
            elif self.tab_index[j] == 1:
                for _ in range(3):
                    num_gd += 1
                    if num_gd == guide_number:
                        guide_group_number = j
                        break
        return guide_group_number

    def check_modes_display(self):
        """
        Display on the interface the last mode that can propagated into a
        squared guide.
        """
        lo = self.doubleSpinBox_lo.value()

        guide_index = self.find_guide_number(
            self.spinBox_guide_nbr_ref.value())
        mode_max = self.bpm.check_modes(
            self.width[guide_index], self.delta_no[guide_index], lo)
        self.mode_number.display(mode_max)

    def addmpl(self, tab='guide', pow_index=0):
        """
        Add the selected plots on the guide, light or compute window.

        Parameters
        ----------
        tab : str, optional
            'guide' or 'light'.
            'guide' by default.
        pow_index : int, optional
            Add the first guide and light step if 0 or the last step if -1.
            Also display the propagation into (x,z) and guide power if -1 is
            choosen.
            0 by default.
        """
        pow_index_guide = pow_index

        if pow_index < 0:
            pow_index_guide -= 1  # Display the -2 guide for the -1 beam

        x_min = self.x[0]
        x_max = self.x[-1]
        if (0 in self.tab_index  # If array of guides
                and 0 not in self.nbr_p and 0 not in self.p  # If guides exists
                and self.peaks.min() >= self.x[0]  # If guides in the windows
                and self.peaks.max() <= self.x[-1]):
            x_min = np.min(self.offset_guide - self.nbr_p*self.p)
            x_max = np.max(self.offset_guide + self.nbr_p*self.p)

        if (1 in self.tab_index  # If curved guides
                and self.peaks.min() >= self.x[0]  # If guides in the windows
                and self.peaks.max() <= self.x[-1]):
            x_min = self.peaks.min() - self.width.max()
            x_max = self.peaks.max() + self.width.max()

        if tab == 'guide':
            fig = Figure()
            fig.set_tight_layout(True)  # Prevent axes to be cut when resizing
            ax1 = fig.add_subplot(111)
            ax1.set_title("Guide shape through propagation")
            ax1.set_xlabel('x (µm)')
            ax1.set_ylabel('z (mm)')

            ax1.set_xlim(x_min, x_max)

            # note that a colormesh pixel is based on 4 points
            ax1.pcolormesh(self.xv,
                           self.zv,
                           self.dn[self.dn_disp],
                           cmap='gray')

            self.canvas_guide_prop = FigureCanvas(fig)
            self.plot_guide.addWidget(self.canvas_guide_prop)

            self.toolbar_guide_prop = NavigationToolbar(self.canvas_guide_prop,
                                                        self.canvas_guide_prop,
                                                        coordinates=True)
            self.plot_guide.addWidget(self.toolbar_guide_prop)

            fig = Figure()
            fig.set_tight_layout(True)
            ax2 = fig.add_subplot(111)
            ax2.set_title("Input index profil")
            ax2.set_xlabel('x (µm)')
            ax2.set_ylabel(r'$\Delta_n$')

            if sum(self.nbr_p) > 0:
                verts = [(self.x[0], 0),
                         *zip(self.x, self.dn[pow_index_guide, :]),
                         (self.x[-1], 0)]
                poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
                ax2.add_patch(poly)

            ax2.set_xlim(x_min, x_max)

            if max(self.dn[0, :]) > max(self.dn[-1, :]):
                ax2.set_ylim(0,
                             max(self.dn[0, :])*1.1 + 1E-20)
            else:
                ax2.set_ylim(0,
                             max(self.dn[-1, :])*1.1 + 1E-20)

            ax2.plot(self.x, self.dn[pow_index_guide], 'k')

            self.canvas_guide_in = FigureCanvas(fig)
            self.plot_guide.addWidget(self.canvas_guide_in)

            self.toolbar_guide_in = NavigationToolbar(self.canvas_guide_in,
                                                      self.canvas_guide_in,
                                                      coordinates=True)
            self.plot_guide.addWidget(self.toolbar_guide_in)

        elif tab == 'light':
            fig = Figure()
            fig.set_tight_layout(True)
            ax1 = fig.add_subplot(111)
            ax1.set_title("Light injection into a guide")
            ax1.set_xlabel('x (µm)')
            ax2 = ax1.twinx()

            for tl in ax1.get_yticklabels():
                tl.set_color('k')

            for tl in ax2.get_yticklabels():
                tl.set_color('#1f77b4')

            ax1.set_ylabel(r'$\Delta_n$')
            ax2.set_ylabel('Irradiance ($GW.cm^{-2}$)')

            if 0 not in self.nbr_p:
                verts = [(self.x[0], 0),
                         *zip(self.x, self.dn[pow_index_guide, :]),
                         (self.x[-1], 0)]
                poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
                ax1.add_patch(poly)

            ax1.set_xlim(x_min, x_max)

            ax1.set_ylim(0,
                         max(1.1*self.dn[pow_index_guide, :]) + 1E-20)

            if max(self.progress_pow[0]) != 0:
                ax2.set_ylim(0, 1.1e-13*max(self.progress_pow[0]))

            ax1.plot(self.x, self.dn[pow_index_guide], 'k')
            ax2.plot(self.x, 1e-13*self.progress_pow[pow_index], '#1f77b4')

            # Display light at the beginning of guides
            if pow_index == 0:
                self.canvas_light = FigureCanvas(fig)
                self.plot_light.addWidget(self.canvas_light)

                self.toolbar_2 = NavigationToolbar(self.canvas_light,
                                                   self.canvas_light,
                                                   coordinates=True)
                self.plot_light.addWidget(self.toolbar_2)

            # Display light at the end of guides
            if pow_index < 0:
                ax1.set_title("Light at the end of guides")
                self.canvas_compute_end = FigureCanvas(fig)
                self.plot_compute.addWidget(self.canvas_compute_end)

                self.toolbar_4 = NavigationToolbar(self.canvas_compute_end,
                                                   self.canvas_compute_end,
                                                   coordinates=True)
                self.plot_compute.addWidget(self.toolbar_4)

                # Display light propagation into guides
                fig = Figure()
                fig.set_tight_layout(True)
                ax1 = fig.add_subplot(111)
                ax1.set_title("Light propagation into guides")
                ax1.set_xlabel('x (µm)')
                ax1.set_ylabel('z (mm)')

                ax2.set_xlim(x_min, x_max)

                ax1.pcolormesh(self.xv,
                               self.zv,
                               1e-13 * self.progress_pow)
                self.canvas_compute_propag = FigureCanvas(fig)
                self.plot_compute.addWidget(self.canvas_compute_propag)

                self.toolbar_3 = NavigationToolbar(self.canvas_compute_propag,
                                                   self.canvas_compute_propag,
                                                   coordinates=True)
                self.plot_compute.addWidget(self.toolbar_3)

                # Display guides power
                if (self.checkBox_check_power.isChecked() and
                        0 not in self.nbr_p and 0 not in self.p):
                    t_power_start = time.process_time()
                    fig = Figure()
#                    fig.set_tight_layout(True)
                    ax1 = fig.add_subplot(111)
                    ax1.set_title("Radiant flux in guides")
                    ax1.set_ylim(0, 1)
                    ax1.set_xlabel('z (mm)')
                    ax1.set_ylabel('Radiant flux (u.a)')

                    style = list(lines.lineStyles.keys())[0:-3]
                    style = style + list(lines.lineMarkers.keys())[0:-16]

                    x_beg = np.zeros((self.peaks.shape[0], self.nbr_z),
                                     dtype=int)
                    x_end = np.zeros((self.peaks.shape[0], self.nbr_z),
                                     dtype=int)
                    P = np.zeros((self.peaks.shape[0], self.nbr_z_disp+1))

                    num_gd = 0
                    for i, n in enumerate(self.nbr_p):

                        if self.tab_index[i] == 0:

                            for _ in range(n):
                                [x_beg[num_gd, :],
                                 x_end[num_gd, :]] = self.bpm.guide_position(
                                     self.peaks, num_gd, self.p[i])
                                num_gd += 1

                        elif self.tab_index[i] == 1:
                            # Choose precision at the end for right guide
                            # and choose safety when guides overlapse
                            if self.peaks[num_gd+2, -1] <= self.x[-1]:
                                # accurate at end but may overlap before
                                p0 = (self.peaks[num_gd+2, -1]
                                      - self.peaks[num_gd+1, -1])

                            else:
                                # accurate in middle but miss evanescente part
                                p0 = self.width[i] * self.distance_factor[i]

                            for j in range(3):
                                [x_beg[num_gd, :],
                                 x_end[num_gd, :]] = self.bpm.guide_position(
                                      self.peaks, num_gd, p0)
                                num_gd += 1

                    for i in range(self.peaks.shape[0]):
                        P[i, :] = self.bpm.power_guide(x_beg[i, :],
                                                       x_end[i, :])
                        # plot each power with a different style (nbr_p < 29)
                        if i < 29:
                            ax1.plot(self.z_disp, P[i, :],
                                     style[i], label='P'+str(i))

                        else:  # to have no error but same style
                            ax1.plot(self.z_disp, P[i, :],
                                     '', label='P'+str(i))

                    self.canvas_5 = FigureCanvas(fig)
                    self.verticalLayout_compute_plot.addWidget(self.canvas_5)

                    self.toolbar_5 = NavigationToolbar(self.canvas_5,
                                                       self.canvas_5,
                                                       coordinates=True)
                    self.verticalLayout_compute_plot.addWidget(self.toolbar_5)
                    if self.peaks.shape[0] > 10:
                        ax1.legend(loc="upper right")  # Fast if many plot
                    else:
                        ax1.legend()  # Best if not too many plot
                    ax1.grid()
                    t_power_end = time.process_time()
                    print('Power time: ', t_power_end-t_power_start)

    def rmmpl(self, tab, pow_index=0):
        """
        Remove the selected plots

        Parameters
        ----------
        tab : str
            'guide' or 'light'.
        pow_index : int, optional
            Remove the first light step if 0 or the last step if -1.
            0 by default.
        """
        if tab == 'guide':
            self.plot_guide.removeWidget(self.canvas_guide_prop)
            self.plot_guide.removeWidget(self.canvas_guide_in)
            self.canvas_guide_prop.close()
            self.canvas_guide_in.close()

            self.plot_guide.removeWidget(self.toolbar_guide_prop)
            self.plot_guide.removeWidget(self.toolbar_guide_in)
            self.toolbar_guide_prop.close()
            self.toolbar_guide_in.close()

        elif tab == 'light':
            if pow_index == 0:
                self.plot_light.removeWidget(self.canvas_light)
                self.canvas_light.close()
                self.plot_light.removeWidget(self.toolbar_2)
                self.toolbar_2.close()

            if pow_index < 0:
                self.plot_compute.removeWidget(self.canvas_compute_propag)
                self.canvas_compute_propag.close()
                self.plot_compute.removeWidget(self.toolbar_3)
                self.toolbar_3.close()

                self.verticalLayout_compute_plot.removeWidget(self.canvas_5)
                self.canvas_5.close()
                self.verticalLayout_compute_plot.removeWidget(self.toolbar_5)
                self.toolbar_5.close()

                self.plot_compute.removeWidget(self.canvas_compute_end)
                self.canvas_compute_end.close()
                self.plot_compute.removeWidget(self.toolbar_4)
                self.toolbar_4.close()

    def save_guide(self, guide_selec=False):
        """
        Save the interface variables into the guides variables.
        """
        # if more than one guide and if no guide selected manualy
        if str(guide_selec) == 'False':
            guide_selec = int(self.comboBox_guide.currentIndex())  # Choice

        self.length_z = self.doubleSpinBox_length_z.value()
        self.dist_z = self.doubleSpinBox_dist_z.value()
        self.nbr_z_disp = self.spinBox_nbr_z_disp.value()
        self.length_x = self.doubleSpinBox_length_x.value()
        self.dist_x = self.doubleSpinBox_dist_x.value()
        self.no = self.doubleSpinBox_n.value()

        self.width[guide_selec] = self.doubleSpinBox_width.value()
        self.offset_guide[
            guide_selec] = self.doubleSpinBox_offset_guide.value()
        self.delta_no[guide_selec] = self.doubleSpinBox_dn.value()
        self.shape_gauss_check[guide_selec] = float(
            self.radioButton_gaussian.isChecked())
        self.gauss_pow[guide_selec] = int(self.spinBox_gauss_pow.value())
        self.shape_squared_check[guide_selec] = float(
            self.radioButton_squared.isChecked())
        self.nbr_p[guide_selec] = self.spinBox_nb_p.value()
        self.p[guide_selec] = self.doubleSpinBox_p.value()
        self.curve[guide_selec] = self.doubleSpinBox_curve.value()
        self.half_delay[guide_selec] = self.doubleSpinBox_half_delay.value()
        self.distance_factor[
            guide_selec] = self.doubleSpinBox_distance_factor.value()

        self.tab_index[
            guide_selec] = self.tabWidget_morphology_guide.currentIndex()
#        print("Guide variables saved")

    def get_guide(self):
        """
        Set the saved values of the guide variables on the interface.
        """
        self.doubleSpinBox_length_z.setValue(self.length_z)
        self.doubleSpinBox_dist_z.setValue(self.dist_z)
        self.spinBox_nbr_z_disp.setValue(self.nbr_z_disp)
        self.doubleSpinBox_length_x.setValue(self.length_x)
        self.doubleSpinBox_dist_x.setValue(self.dist_x)
        self.doubleSpinBox_n.setValue(self.no)

        guide_selec = int(self.comboBox_guide.currentIndex())  # choice

        if self.previous_guide != guide_selec:
            self.save_guide(self.previous_guide)

        if self.comboBox_guide.count() >= 1:  # if more than one beams
            guide_selec = int(self.comboBox_guide.currentIndex())  # choice
        else:  # Not supposed to happen
            raise ValueError("Can't have no guide variables")

        self.doubleSpinBox_width.setValue(self.width[guide_selec])
        self.doubleSpinBox_offset_guide.setValue(
            self.offset_guide[guide_selec])
        self.doubleSpinBox_dn.setValue(self.delta_no[guide_selec])
        self.radioButton_gaussian.setChecked(
            self.shape_gauss_check[guide_selec])
        self.spinBox_gauss_pow.setValue(self.gauss_pow[guide_selec])
        self.radioButton_squared.setChecked(
            self.shape_squared_check[guide_selec])
        self.spinBox_nb_p.setValue(self.nbr_p[guide_selec])
        self.doubleSpinBox_p.setValue(self.p[guide_selec])
        self.doubleSpinBox_curve.setValue(self.curve[guide_selec])
        self.doubleSpinBox_half_delay.setValue(self.half_delay[guide_selec])
        self.doubleSpinBox_distance_factor.setValue(
            self.distance_factor[guide_selec])
        self.tabWidget_morphology_guide.setCurrentIndex(
            self.tab_index[guide_selec])
        self.spinBox_gauss_pow.setEnabled(self.shape_gauss_check[guide_selec])

        self.previous_guide = guide_selec  # Save the # of the current guide

    def save_light(self, beam_selec=False):
        """
        Save the interface variables into the lights variables.

        Parameters
        ----------
        beam_selec: int, bool, optional
            Number of the beam to save into the variables.
            False by default to get the currently displayed beam.
        """
        self.lo = self.doubleSpinBox_lo.value()
        self.theta_ext = self.doubleSpinBox_theta_ext.value()

        # if more than one beams and if no beams selected manualy
        if str(beam_selec) == 'False':
            beam_selec = int(self.comboBox_light.currentIndex())  # Choice

        self.fwhm[beam_selec] = self.doubleSpinBox_fwhm.value()
        self.offset_light[beam_selec] = self.doubleSpinBox_offset_light.value()
        self.irrad[beam_selec] = self.doubleSpinBox_intensity_light.value()
        self.mode[beam_selec] = self.spinBox_mode.value()
        self.mode_guide_ref[beam_selec] = self.spinBox_guide_nbr_ref.value()
        self.offset_check[beam_selec] = (
            self.checkBox_offset_light.isChecked())
        self.offset_light_peak[beam_selec] = (
            self.spinBox_offset_light_peak.value())
        self.gaussian_check[beam_selec] = (
            self.radioButton_gaussian_light.isChecked())
        self.square_check[beam_selec] = (
            self.radioButton_squared_light.isChecked())
        self.mode_check[beam_selec] = self.radioButton_mode.isChecked()
        self.all_modes_check[beam_selec] = (
            self.radioButton_all_modes.isChecked())
        self.airy_check[beam_selec] = (
            self.radioButton_airy.isChecked())
        self.airy_zero[beam_selec] = self.spinBox_airy_zero.value()
        self.lobe_size[beam_selec] = self.doubleSpinBox_lobe_size.value()

    def get_light(self):
        """
        Set the saved values of the light variables on the interface.
        """
        beam_selec = int(self.comboBox_light.currentIndex())  # choice

        if self.previous_beam != beam_selec:
            self.save_light(self.previous_beam)

        self.doubleSpinBox_lo.setValue(self.lo)
        self.doubleSpinBox_theta_ext.setValue(self.theta_ext)

        if self.comboBox_light.count() >= 1:  # if more than one beams
            beam_selec = int(self.comboBox_light.currentIndex())  # choice
        else:  # Not supposed to happen
            raise ValueError("Can't have no beam variables")

        self.doubleSpinBox_fwhm.setValue(self.fwhm[beam_selec])
        self.doubleSpinBox_offset_light.setValue(
            self.offset_light[beam_selec])
        self.doubleSpinBox_intensity_light.setValue(self.irrad[beam_selec])
        self.spinBox_mode.setValue(self.mode[beam_selec])
        self.spinBox_guide_nbr_ref.setValue(self.mode_guide_ref[beam_selec])
        self.checkBox_offset_light.setChecked(self.offset_check[beam_selec])
        self.spinBox_offset_light_peak.setValue(
            self.offset_light_peak[beam_selec])
        self.radioButton_gaussian_light.setChecked(
            self.gaussian_check[beam_selec])
        self.radioButton_squared_light.setChecked(
            self.square_check[beam_selec])
        self.radioButton_mode.setChecked(self.mode_check[beam_selec])
        self.radioButton_all_modes.setChecked(self.all_modes_check[beam_selec])
        self.radioButton_airy.setChecked(self.airy_check[beam_selec])
        self.spinBox_airy_zero.setValue(self.airy_zero[beam_selec])
        self.doubleSpinBox_lobe_size.setValue(self.lobe_size[beam_selec])
        # if checkBox_offset_light checked then activate
        self.spinBox_offset_light_peak.setEnabled(
            self.offset_check[beam_selec])
        self.doubleSpinBox_offset_light.setDisabled(
            self.offset_check[beam_selec])
        self.spinBox_airy_zero.setEnabled(self.airy_check[beam_selec])
        self.doubleSpinBox_lobe_size.setEnabled(self.airy_check[beam_selec])
        self.spinBox_mode.setEnabled(self.mode_check[beam_selec])
        self.spinBox_guide_nbr_ref.setEnabled(self.mode_check[beam_selec])

        self.previous_beam = beam_selec  # Save the number of the current beam

    def save_compute(self):
        """
        Save the interface variables into the compute variables.
        """
        self.lost_check = float(self.checkBox_lost.isChecked())
        self.guide_lost = np.array(
            [self.spinBox_guide_lost.value()], dtype=int)
        self.width_lost = np.array([self.doubleSpinBox_width_lost.value()])
        self.alpha = self.doubleSpinBox_lost.value()

        self.kerr_check = float(self.checkBox_kerr.isChecked())
        self.kerr_loop = self.spinBox_kerr_loop.value()
        self.chi3 = self.doubleSpinBox_chi3.value()
        self.variance_check = float(self.checkBox_variance.isChecked())
        self.power_check = float(self.checkBox_check_power.isChecked())
#        print("Compute variables saved")

    def get_compute(self):
        """
        Set the saved values of the compute variables on the interface.
        """
        self.checkBox_lost.setChecked(self.lost_check)
        self.frame_lost.setEnabled(self.lost_check)
        self.spinBox_guide_lost.setValue(self.guide_lost[0])
        self.doubleSpinBox_width_lost.setValue(self.width_lost[0])
        self.doubleSpinBox_lost.setValue(self.alpha)

        self.checkBox_kerr.setChecked(self.kerr_check)
        self.spinBox_kerr_loop.setValue(self.kerr_loop)
        self.doubleSpinBox_chi3.setValue(self.chi3)
        self.frame_kerr.setEnabled(self.kerr_check)
        self.checkBox_variance.setEnabled(self.kerr_check)
        self.checkBox_variance.setChecked(self.variance_check)
        if not self.kerr_check:
            self.checkBox_variance.setChecked(False)
        self.checkBox_check_power.setChecked(self.power_check)

    @pyqtSlot()
    def on_click_guide(self):
        """
        Compute and displayed the waguides.
        """
#        print('button click guide array')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.rmmpl('guide')
        self.rmmpl('light')
        self.calculate_guide()
        self.calculate_light()
        self.addmpl('guide')
        self.addmpl('light')
        QApplication.restoreOverrideCursor()
        self.show_estimate_time()

    @pyqtSlot()
    def on_click_light(self):
        """
        Compute the light and display it with guides.
        """
#        print('button click light')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.rmmpl(tab='light')
        self.calculate_light()
        self.addmpl(tab='light')
        QApplication.restoreOverrideCursor()
        self.show_estimate_time()

    @pyqtSlot()
    def on_click_compute(self):
        """
        Compute the propagation using the guide and light informations.
        """
#        print('button click compute')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.show_estimate_time()

        if max(self.progress_pow[0]) != 0:

            self.rmmpl(tab='light', pow_index=-1)

            if not self.calculate_guide_done:
                self.rmmpl('guide')
                self.calculate_guide()
                self.addmpl('guide')
                self.rmmpl(tab='light')
                self.calculate_light()
                self.addmpl(tab='light')

            self.calculate_propagation()
            self.addmpl(tab='light', pow_index=-1)

        else:
            print("no light to compute")

        QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def on_click_create_guide(self):
        """Create a new guide with the displayed variables.
        """
        width = self.doubleSpinBox_width.value()
        offset_guide = self.doubleSpinBox_offset_guide.value()
        delta_no = self.doubleSpinBox_dn.value()
        shape_gauss_check = self.radioButton_gaussian.isChecked()
        gauss_pow = int(self.spinBox_gauss_pow.value())
        shape_squared_check = self.radioButton_squared.isChecked()
        nbr_p = self.spinBox_nb_p.value()
        p = self.doubleSpinBox_p.value()
        curve = self.doubleSpinBox_curve.value()
        half_delay = self.doubleSpinBox_half_delay.value()
        distance_factor = self.doubleSpinBox_distance_factor.value()
        tab_index = self.tabWidget_morphology_guide.currentIndex()

        self.width = np.append(self.width, width)
        self.offset_guide = np.append(self.offset_guide, offset_guide)
        self.delta_no = np.append(self.delta_no, delta_no)
        self.shape_gauss_check = np.append(self.shape_gauss_check,
                                           shape_gauss_check)
        self.gauss_pow = np.append(self.gauss_pow, gauss_pow)
        self.shape_squared_check = np.append(self.shape_squared_check,
                                             shape_squared_check)
        self.nbr_p = np.append(self.nbr_p, nbr_p)
        self.p = np.append(self.p, p)
        self.curve = np.append(self.curve, curve)
        self.half_delay = np.append(self.half_delay, half_delay)
        self.distance_factor = np.append(self.distance_factor, distance_factor)
        self.tab_index = np.append(self.tab_index, tab_index)

        nbr_guide = self.comboBox_guide.count()  # how many item left
        self.comboBox_guide.addItem("Guide "+str(nbr_guide+1))  # add new index
        self.comboBox_guide.setCurrentIndex(nbr_guide)  # show new index
        self.previous_beam = nbr_guide  # Change the current selected guide

    @pyqtSlot()
    def on_click_create_light(self):
        """Create a new beam with the displayed variables.
        """
        fwhm = self.doubleSpinBox_fwhm.value()
        offset_light = self.doubleSpinBox_offset_light.value()
        irrad = self.doubleSpinBox_intensity_light.value()
        offset_check = self.checkBox_offset_light.isChecked()
        gaussian_check = self.radioButton_gaussian_light.isChecked()
        square_check = self.radioButton_squared_light.isChecked()
        mode_check = self.radioButton_mode.isChecked()
        all_modes_check = self.radioButton_all_modes.isChecked()
        mode = self.spinBox_mode.value()
        mode_guide_ref = self.spinBox_guide_nbr_ref.value()
        offset_light_peak = self.spinBox_offset_light_peak.value()
        airy_check = self.radioButton_airy.isChecked()
        airy_zero = self.spinBox_airy_zero.value()
        lobe_size = self.doubleSpinBox_lobe_size.value()

        self.fwhm = np.append(self.fwhm, fwhm)
        self.offset_light = np.append(self.offset_light, offset_light)
        self.irrad = np.append(self.irrad, irrad)
        self.mode = np.append(self.mode, mode)
        self.mode_guide_ref = np.append(self.mode_guide_ref, mode_guide_ref)
        self.offset_check = np.append(self.offset_check, offset_check)
        self.offset_light_peak = np.append(self.offset_light_peak,
                                           offset_light_peak)
        self.gaussian_check = np.append(self.gaussian_check, gaussian_check)
        self.square_check = np.append(self.square_check, square_check)
        self.mode_check = np.append(self.mode_check, mode_check)
        self.all_modes_check = np.append(self.all_modes_check, all_modes_check)
        self.airy_check = np.append(self.airy_check, airy_check)
        self.airy_zero = np.append(self.airy_zero, airy_zero)
        self.lobe_size = np.append(self.lobe_size, lobe_size)

        nbr_light = self.comboBox_light.count()  # how many item left
        self.comboBox_light.addItem("Beam "+str(nbr_light+1))  # add new index
        self.comboBox_light.setCurrentIndex(nbr_light)  # show new index
        self.previous_beam = nbr_light  # Change the current selected beam

    @pyqtSlot()
    def on_click_delete_guide(self):
        """
        Delete the current displayed guide and displayed the next one.
        """
        nbr_guide = self.comboBox_guide.count()

        if nbr_guide > 1:  # Can't delete if remains only 1 guide
            guide_selec = int(self.comboBox_guide.currentIndex())  # choice
            self.width = np.delete(self.width, guide_selec)
            self.offset_guide = np.delete(self.offset_guide, guide_selec)
            self.delta_no = np.delete(self.delta_no, guide_selec)
            self.shape_gauss_check = np.delete(self.shape_gauss_check,
                                               guide_selec)
            self.gauss_pow = np.delete(self.gauss_pow, guide_selec)
            self.shape_squared_check = np.delete(self.shape_squared_check,
                                                 guide_selec)
            self.nbr_p = np.delete(self.nbr_p, guide_selec)
            self.p = np.delete(self.p, guide_selec)
            self.curve = np.delete(self.curve, guide_selec)
            self.half_delay = np.delete(self.half_delay, guide_selec)
            self.distance_factor = np.delete(self.distance_factor, guide_selec)

            nbr_guide -= 1

            self.comboBox_guide.clear()  # remove all beams number

            for i in range(nbr_guide):  # create again with new number
                self.comboBox_guide.addItem("Guide "+str(i+1))

            # set same guide index if not the last else reduce the index by 1
            if guide_selec == nbr_guide and guide_selec != 0:
                guide_selec -= 1

            self.comboBox_guide.setCurrentIndex(guide_selec)
            self.previous_guide = guide_selec  # Change the selected guide
            self.get_guide()  # Display values of the previous or next guide

    @pyqtSlot()
    def on_click_delete_light(self):
        """
        Delete the current displayed beam and displayed the next one.
        """
        nbr_light = self.comboBox_light.count()

        if nbr_light > 1:  # Can't delete if remains only 1 beam
            beam_selec = int(self.comboBox_light.currentIndex())  # choice
            self.fwhm = np.delete(self.fwhm, beam_selec)
            self.offset_light = np.delete(self.offset_light, beam_selec)
            self.irrad = np.delete(self.irrad, beam_selec)
            self.mode = np.delete(self.mode, beam_selec)
            self.mode_guide_ref = np.delete(self.mode_guide_ref, beam_selec)
            self.offset_check = np.delete(self.offset_check, beam_selec)
            self.offset_light_peak = np.delete(self.offset_light_peak,
                                               beam_selec)
            self.gaussian_check = np.delete(self.gaussian_check, beam_selec)
            self.square_check = np.delete(self.square_check, beam_selec)
            self.mode_check = np.delete(self.mode_check, beam_selec)
            self.all_modes_check = np.delete(self.all_modes_check,
                                             beam_selec)
            self.airy_check = np.delete(self.airy_check, beam_selec)
            self.airy_zero = np.delete(self.airy_zero, beam_selec)
            self.lobe_size = np.delete(self.lobe_size, beam_selec)

            nbr_light -= 1

            self.comboBox_light.clear()  # remove all beams number

            for i in range(nbr_light):  # create again with new number
                self.comboBox_light.addItem("Beam "+str(i+1))

            # set same beam index if not the last else reduce the index by 1
            if beam_selec == nbr_light and beam_selec != 0:
                beam_selec -= 1

            self.comboBox_light.setCurrentIndex(beam_selec)
            self.previous_beam = beam_selec  # Change the current selected beam
            self.get_light()  # Display values of the previous or next beam

    @pyqtSlot()
    def open_file_name(self):
        """
        Open a dialog window to select the file to open, and call
        :meth:`open_file` to open the file.

        Source: https://pythonspot.com/pyqt5-file-dialog/

        Notes
        -----
        This method has a try/except implemented to check if the openned file
        contains all the variables.
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Import data",
                                                  "",
                                                  "Text Files (*.txt)",
                                                  options=options)
        if filename:
            try:
                self.open_file(filename)
                self.filename = filename  # Next save will overwrite this file
                self.setWindowTitle("Beampy - "+filename)
            except KeyError as ex:
                print("missing variable", ex, "in the file.")
                print("Add it manually to remove this error.")
#            except Exception as ex:  # This execption was removed to see the
                # # error location. The program won't crash thanks to try
#                print("Unknown error when openning the file:", filename)
#                print("Error:", ex)

    def open_file(self, filename):
        """
        Set guides, beams and computes variables from a choosen file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
#        https://www.tutorialspoint.com/
#        How-to-create-a-Python-dictionary-from-text-file
        dico = {}
        f = open(filename, 'r')
        for line in f:
            (variables, *val) = line.split()  # Assume: variable_name values
#            print(variables, val)
            dico[str(variables)] = val
        f.close()

        # Save the displayed values (only useful if the displayed variable is
#       not in the openned file)
        self.save_guide()
        self.save_light()
        self.save_compute()

        # Guide variables
#        if dico.get('length_z') is not None: # TODO: if want to continue
#        without the variable and choose the displayed values instead
        self.length_z = float(dico['length_z'][0])
        self.dist_z = float(dico['dist_z'][0])
        self.nbr_z_disp = int(dico['nbr_z_disp'][0])
        self.length_x = float(dico['length_x'][0])
        self.dist_x = float(dico['dist_x'][0])
        self.no = float(dico['no'][0])

        self.width = np.array(dico['width'], dtype=float)
        self.offset_guide = np.array(dico['offset_guide'], dtype=float)
        self.delta_no = np.array(dico['delta_no'], dtype=float)
        self.shape_gauss_check = np.array(dico['shape_gauss_check'],
                                          dtype=float)
        self.gauss_pow = np.array(dico['gauss_pow'], dtype=int)
        self.shape_squared_check = np.array(dico['shape_squared_check'],
                                            dtype=float)
        self.nbr_p = np.array(dico['nbr_p'], dtype=int)
        self.p = np.array(dico['p'], dtype=float)
        self.curve = np.array(dico['curve'], dtype=float)
        self.half_delay = np.array(dico['half_delay'], dtype=float)
        self.distance_factor = np.array(dico['distance_factor'], dtype=float)
        self.tab_index = np.array(dico['tab_index'], dtype=float)

        # Light variables
        self.lo = float(dico['lo'][0])
        self.theta_ext = float(dico['theta_ext'][0])

        self.fwhm = np.array(dico['fwhm'], dtype=float)
        self.offset_light = np.array(dico['offset_light'], dtype=float)
        self.irrad = np.array(dico['irrad'], dtype=float)
        self.offset_check = np.array(dico['offset_check'], dtype=float)
        self.gaussian_check = np.array(dico['gaussian_check'], dtype=float)
        self.square_check = np.array(dico['square_check'], dtype=float)
        self.mode_check = np.array(dico['mode_check'], dtype=float)
        self.all_modes_check = np.array(dico['all_modes_check'], dtype=float)
        self.mode = np.array(dico['mode'], dtype=int)
        self.mode_guide_ref = np.array(dico['mode_guide_ref'], dtype=int)
        self.offset_light_peak = np.array(
            dico['offset_light_peak'], dtype=int)
        self.airy_check = np.array(dico['airy_check'], dtype=float)
        self.airy_zero = np.array(dico['airy_zero'], dtype=int)
        self.lobe_size = np.array(dico['lobe_size'], dtype=float)

        # Compute variables
        self.lost_check = float(dico['lost_check'][0])
        self.guide_lost = np.array(dico['guide_lost'], dtype=int)
        self.width_lost = np.array(dico['width_lost'], dtype=float)
        self.alpha = float(dico['alpha'][0])
        self.kerr_check = float(dico['kerr_check'][0])
        self.kerr_loop = int(dico['kerr_loop'][0])
        self.chi3 = float(dico['chi3'][0])
        self.variance_check = float(dico['variance_check'][0])
        self.power_check = float(dico['power_check'][0])

        nbr_guide = len(self.width)
        self.comboBox_guide.clear()  # Remove all guides number

        nbr_light = len(self.fwhm)
        self.comboBox_light.clear()  # Remove all beams number

        for i in range(nbr_guide):  # Create again with new number
            self.comboBox_guide.addItem("Guide "+str(i+1))

        for i in range(nbr_light):  # Create again with new number
            self.comboBox_light.addItem("Beam "+str(i+1))

        self.previous_guide = 0  # Change the current selected guide
        self.previous_beam = 0  # Change the current selected beam

        self.get_guide()  # Set guides boxes value
        self.get_light()  # Set lights boxes value
        self.get_compute()  # Set compute boxes value

        self.on_click_guide()

        print("file openned")

    @pyqtSlot()
    def save_quick(self):
        """
        Check if a file is already selected and if so, save into it.
        Else, call the :meth:`save_file_name` to ask a filename.
        """
        if self.filename is None:
            self.save_file_name()
        else:
            self.save_file(self.filename)

    def save_file_name(self):
        """
        Open a dialog window to select the saved file name and call
        :meth:`save_file` to save the file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save data",
                                                  "",
                                                  "Text Files (*.txt)",
                                                  options=options)
        if filename:
            if filename[-4:] != '.txt':
                filename = filename + '.txt'
            self.filename = filename
            self.save_file(filename)
            self.setWindowTitle("Beampy - "+filename)

    def save_file(self, filename):
        """
        Save guides, beams and computes variables into a choosen file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        self.save_guide()
        self.save_light()
        self.save_compute()

        f = open(filename, "w")

        # Guide variables
        f.write('length_z ' + str(self.length_z) + '\n')
        f.write('dist_z ' + str(self.dist_z) + '\n')
        f.write('nbr_z_disp ' + str(self.nbr_z_disp) + '\n')
        f.write('length_x ' + str(self.length_x) + '\n')
        f.write('dist_x ' + str(self.dist_x) + '\n')
        f.write('no ' + str(self.no) + '\n')

        f.write('width ' + str(self.width).replace("[", "").replace("]", "")
                + '\n')
        f.write('offset_guide ' + str(
            self.offset_guide).replace("[", "").replace("]", "")
                + '\n')
        f.write('delta_no ' + str(
            self.delta_no).replace("[", "").replace("]", "")
                + '\n')
        f.write('shape_gauss_check ' + str(
            self.shape_gauss_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('gauss_pow ' + str(
            self.gauss_pow).replace("[", "").replace("]", "")
                + '\n')
        f.write('shape_squared_check ' + str(
            self.shape_squared_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('nbr_p ' + str(self.nbr_p).replace("[", "").replace("]", "")
                + '\n')
        f.write('p ' + str(self.p).replace("[", "").replace("]", "")
                + '\n')
        f.write('curve ' + str(self.curve).replace("[", "").replace("]", "")
                + '\n')
        f.write('half_delay ' + str(
            self.half_delay).replace("[", "").replace("]", "")
                + '\n')
        f.write('distance_factor ' + str(
            self.distance_factor).replace("[", "").replace("]", "")
                + '\n')
        f.write('tab_index ' + str(
            self.tab_index).replace("[", "").replace("]", "")
                + '\n')

        # light variables
        f.write('lo ' + str(self.lo) + '\n')
        f.write('theta_ext ' + str(self.theta_ext) + '\n')

        f.write('fwhm '
                + str(self.fwhm).replace("[", "").replace("]", "")
                + '\n')
        f.write('offset_light '
                + str(self.offset_light).replace("[", "").replace("]", "")
                + '\n')
        f.write('irrad '
                + str(self.irrad).replace("[", "").replace("]", "")
                + '\n')
        f.write('offset_check '
                + str(self.offset_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('gaussian_check '
                + str(self.gaussian_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('square_check '
                + str(self.square_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('mode_check '
                + str(self.mode_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('all_modes_check '
                + str(self.all_modes_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('mode '
                + str(self.mode).replace("[", "").replace("]", "")
                + '\n')
        f.write('mode_guide_ref '
                + str(self.mode_guide_ref).replace("[", "").replace("]", "")
                + '\n')
        f.write('offset_light_peak '
                + str(self.offset_light_peak).replace("[", "").replace("]", "")
                + '\n')
        f.write('airy_check '
                + str(self.airy_check).replace("[", "").replace("]", "")
                + '\n')
        f.write('airy_zero '
                + str(self.airy_zero).replace("[", "").replace("]", "")
                + '\n')
        f.write('lobe_size '
                + str(self.lobe_size).replace("[", "").replace("]", "")
                + '\n')

        # compute variables
        f.write('lost_check ' + str(self.lost_check) + '\n')
        f.write('guide_lost '
                + str(self.guide_lost).replace("[", "").replace("]", "")
                + '\n')
        f.write('width_lost '
                + str(self.width_lost).replace("[", "").replace("]", "")
                + '\n')
        f.write('alpha ' + str(self.alpha) + '\n')
        f.write('kerr_check ' + str(self.kerr_check) + '\n')
        f.write('kerr_loop ' + str(self.kerr_loop) + '\n')
        f.write('chi3 ' + str(self.chi3) + '\n')
        f.write('variance_check ' + str(self.variance_check) + '\n')
        f.write('power_check ' + str(self.power_check) + '\n')
        f.close()
        print("file saved")


def open_doc():
    """
    Function that open the local html documentation - describing the beampy
    modules - if exist, or the online version otherwise.
    """
    file = __file__  # Module name
    # Replaces characters only when called from outer files
    file = file.replace("\\", "/")
    file = file.split("/")
    file = file[:-2]  # Remove the folder and file name

    file2 = str()
    for line in file:
        file2 = file2+"/"+line

    file = file2[1:]+"/docs/html/index.html"
    exists = os.path.isfile(file)

    if exists:
        webbrowser.open(file, new=2)  # Open file in a new tab (new=2)
    else:
        print("The documentation can't be found localy in:", file)
        file = "https://beampy.readthedocs.io"
        print("Openning the online version at:", file)
        webbrowser.open(file, new=2)  # Open file in a new tab (new=2)


def open_app():
    """
    Function used to open the app.
    Can be called directly from beampy.
    """
    app = QApplication(sys.argv)  # Define the app
    myapp = UserInterface()  # Run the app
    myapp.show()  # Show the form
    app.exec_()  # Execute the app in a loop


if __name__ == "__main__":
    open_app()
