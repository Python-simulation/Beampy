"""
The bpm module contain the Bpm class used to simulate the light propagation -
within low refractive index variation
and small angle (paraxial approximation) -
using the Beam Propagation Method.

This module was done by Jonathan Peltier during a master
university course from the PAIP master of the université de Lorraine,
under the directive of Pr. Nicolas Fressengeas.

The bpm codes are mainly based on a compilation of MatLab codes initialy
developed by Régis Grasser during his PhD thesis[2],
and later modified at the FEMTO-ST institute of the Université de
Franche-Comté and at the LMOPS laboratory [3] of the
Université de Lorraine.

[1] K. Okamoto, in Fundamentals of Optical Waveguides,
2nd ed., edited by K. Okamoto (Academic, Burlington, 2006), pp. 329–397.

[2] "Generation et propagation de reseaux periodiques de solitons spatiaux
dans un milieu de kerr massif" PhD thesis, université de Franche-Comté 1998.

[3] H. Oukraou et. al., Broadband photonic transport between waveguides by
adiabatic elimination Phys. Rev. A, 97 023811 (2018).
"""

from math import pi, ceil, radians, sqrt, log, sin, cos, acos, asin, exp
import time
from scipy import special
from numpy.fft import fft, ifft, fftshift
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numba


@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def abs2(x):
    """Square modulus of x. Fastest way possible for a numpy array."""
    return x.real**2 + x.imag**2


class Bpm():
    """
    The Bpm class is used to simulate light propagation -
    within small refractive index variation guides
    and small angle of propagation (paraxial) -
    using the Beam Propagation Method.

    Parameters
    ----------
    no : float
        Refractive index of the cladding.
    lo : float
        Wavelength of the beam in vaccum (µm).
    length_z : float
        Size of the compute window over z (µm).
    dist_z : float
        Step over z (µm)
    nbr_z_disp : int
        Number of points to display over z.
    length_x : float
        Size of the compute window over x (µm).
    dist_x : float
        Step over x (µm)
    """

    def __init__(self, no, lo,
                 length_z, dist_z, nbr_z_disp,
                 length_x, dist_x):
        """
        The Bpm class is used to simulate light propagation -
        within small refractive index variation guides
        and small angle of propagation (paraxial) -
        using the Beam Propagation Method.

        Parameters
        ----------
        no : float
            Refractive index of the cladding
        lo : float
            Wavelength of the beam in vaccum (µm).
        length_z : float
            Size of the compute window over z (µm).
        dist_z : float
            Step over z (µm).
        nbr_z_disp : int
            Number of points to display over z.
        length_x : float
            Size of the compute window over x (µm).
        dist_x : float
            Step over x (µm).

        Notes
        -----
        This method creates the following variables within the class
        :class:`Bpm`:

        - All input variables.
        - ko: the free space vector (1/µm).
        """
        self.no = no
        self.lo = lo
        self.ko = 2*pi / self.lo  # linear wave vector in free space (1/µm)
        self.length_z = length_z
        self.dist_z = dist_z
        self.nbr_z_disp = nbr_z_disp
        self.dist_x = dist_x
        self.length_x = length_x

    def create_x_z(self):
        """
        Create the x, z array and ajust the resolution variables.

        Returns
        -------
        length_z : float
            Corrected value due to nbr_z being an int.
        nbr_z : int
            Number of points computed over z.
        nbr_z_disp : int
            Corrected value due to pas being an int.
        length_x : float
            Corrected value due to nbr_x being an int.
        nbr_x : int
            Number of point over x (µm).
        x : array
            x values between [-length_x/2, length_x/2-dist_x] center on 0.

        Notes
        -----
        This method creates the following variables within the class
        :class:`Bpm`:

        - pas : Interval of computed points between each displayed points.
        """
        assert self.nbr_z_disp > 0

        self.nbr_z = ceil(self.length_z / self.dist_z)
        self.length_z = self.nbr_z * self.dist_z
        self.pas = ceil(self.length_z / (self.nbr_z_disp * self.dist_z))
        self.nbr_z_disp = ceil(self.length_z / (self.pas * self.dist_z))
        self.nbr_z_disp += 1  # add 1 for the initial field
        self.nbr_z += 1  # add 1 for the initial field
        self.nbr_x = ceil(self.length_x / self.dist_x)  # nbr points over x

        # check if even number
        if self.nbr_x % 2 != 0:
            self.nbr_x += 1

        # check if multiple of 8: speeds up execution
        # (was also needed for a obsolete feature)
        for _ in range(3):
            if self.nbr_x % 8 != 0:
                self.nbr_x += 2
            else:
                break

        self.length_x = self.nbr_x * self.dist_x
        self.x = np.linspace(-self.length_x/2,
                             self.length_x/2 - self.dist_x,
                             self.nbr_x)

        return [self.length_z, self.nbr_z, self.nbr_z_disp-1,
                self.length_x, self.nbr_x, self.x]

    # Guides #

    def squared_guide(self, width):
        """
        A lambda function than returns a centered rectangular shape.

        return 1 if :math:`x >= -width/2` and :math:`x <= width/2`
        else return 0.

        Parameters
        ----------
        width : float
            Waveguide width.
        """
        return lambda x: (x >= -width/2) & (x <= width/2)

    def gauss_guide(self, width, gauss_pow=1):
        """
        A lambda function than return a centered super-Gaussian shape.

        :math:`e^{-(x/w)^{2P}}`

        The waist is defined as width/2 and correspond to the 1/e
        relative value.

        See :func:`.example_guides_x` for more details.

        Parameters
        ----------
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        gauss_pow : int, optional
            Index of the super-gaussian guide with 1 being a regural gaussian
            guide and 4 being the conventionnal super-gaussian guide used to
            describe realistic waveguides.
            See on en.wikipedia.org/wiki/Gaussian_function
            #Higher-order_Gaussian_or_super-Gaussian_function.
            1 by Default.
        """
        if width == 0:
            return lambda x: 0
        w = width / 2  # width is diameter and w is radius
        return lambda x: np.exp(-(x / w)**(2*gauss_pow))

    def create_guides(self, shape, delta_no, nbr_p, p, offset_guide=0, z=0):
        """
        Create an array of guides over x using peaks positions and for a given
        shape.

        Parameters
        ----------
        shape : method
            :meth:`squared_guide`, :meth:`gauss_guide` or any lambda function
            that takes one argument and return the relative refractive index
            for the input position.
        delta_no : float
            Difference of refractive index between the core and the cladding.
            Can contain the losses throught the imaginary part.
        nbr_p : int
            Number of guides.
        p : float
            Distance between two guides center (µm).
        offset_guide : float, optional
            Guide offset from the center (µm). 0 by default.
        z : list
            list [start, end] defining the waveguide length. Default length=
            windows length.

        Returns
        -------
        peaks : array-like
            Central position of each guide [guide,z].
        dn : array-like
            Difference of refractive index [z,x]. Can contain the losses
            throught the imaginary part.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_z, nbr_x, x, dist_x.
        """
        peaks = np.array([[None]*self.nbr_z]*nbr_p)
        dn = np.zeros((self.nbr_z, self.nbr_x))
        dn_z = np.zeros(self.nbr_x)

        if nbr_p == 0:
            return [np.array([[None]*self.nbr_z]), dn]
        peaks_z = (p*np.linspace(-nbr_p/2, nbr_p/2-1, nbr_p)
                   + p/2
                   + offset_guide)
        dn_fix = shape(self.x)  # guide shape center on 0

        # Sum each identical guide with an offset defined by peaks_z
        for i in range(nbr_p):
            dn_z += np.roll(dn_fix, int(round(peaks_z[i] / self.dist_x)))

        if z == 0:
            start = 0
            end = self.nbr_z
        else:
            # assert z[0] >= 0 and z[1] <= self.length_z and z[0] <= z[1]
            if z[0] > z[1]:
                print("Warning, the waveguide beginning occurs after the end.",
                      z[0], "should be <=", z[1])
            if z[1] > self.length_z:
                z[1] = self.length_z

            start = int(z[0]/self.dist_z)
            end = int(z[1]/self.dist_z)

        dn[start:end] = dn_z
        for i in range(start, end):
            peaks[:, i] = peaks_z

        dn = dn*delta_no  # give a value to the shape
        return [peaks, dn]

    def create_curved_guides(self, shape, width, delta_no, curve, half_delay,
                             distance_factor, offset_guide=0):
        """
        Create two curved guides and one linear guide on the center (STIRAP).

        The central positions over x and z are defined as follow:

        Left guide: :math:`x_0-p_{min}-curve(z-length\_z/2-half\_delay)^2`

        Right guide: :math:`x_0+p_{min}+curve(z-length\_z/2+half\_delay)^2`

        Central guide: :math:`x_0`

        Parameters
        ----------
        shape : method
            :meth:`square` or :meth:`gauss`
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        delta_no : float
            Difference of refractive index between the core and the cladding.
            Can contain the losses throught the imaginary part.
        curve : float
            curvature factor in :math:`10^{-8} µm^{-2}`.
        half_delay : float
            Half distance over z in µm bewteen the two external guides where
            they are the closest.
            In other words, the distance from the center and the minimum of one
            of the curved guides over z.
        distance_factor : float
            Factor between the guide width and the minimal distance between the
            two guides =p_min/width.
            If distance_factor=1, the curved guides will touch the central
            guide (p_min=width).
        offset_guide : float, optional
            Waveguide offset from the center (µm). 0 by default.

        Returns
        -------
        peaks : array
            Central position of each guide as peaks[guide,z].
        dn : array
            Difference of refractive index as dn[z,x]. Can contain the losses
            throught the imaginary part.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        length_z, nbr_z, nbr_x, x, dist_x.
        """
        # all points over z
        z = np.linspace(0, self.length_z, self.nbr_z)

        # left curved guide
        sa = (- offset_guide
              + curve*(z - self.length_z/2 - half_delay)**2
              + width*distance_factor)

        # right curved guide
        sb = (offset_guide
              + curve*(z - self.length_z/2 + half_delay)**2
              + width*distance_factor)

        peaks = np.array([-sa,
                          np.array([offset_guide] * self.nbr_z),
                          sb])

        dn = np.zeros((self.nbr_z, self.nbr_x))
        dn_fix = shape(self.x)   # guide shape center on 0

        for i in range(self.nbr_z):
            dn[i, :] = np.roll(dn_fix, int(round(-sa[i] / self.dist_x))) \
                + np.roll(dn_fix, int(round(offset_guide / self.dist_x))) \
                + np.roll(dn_fix, int(round(sb[i] / self.dist_x)))

        dn = dn * delta_no  # give a value to the shape
        return [peaks, dn]

    # Light #

    def gauss_light(self, fwhm, offset_light=0):
        """
        Create a gaussien beam in amplitude.

        :math:`E = e^{-((x-x_0)/w)^{2P}}`

        The waist is defined as fwhm/sqrt(2*log(2)) and correspond to the 1/e
        field value and 1/:math:`e^2` intensity value.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum (for intensity not amplitude) (µm).
        offset_light : float, optional
            Light offset from center in µm. 0 by default.

        Returns
        -------
        field : array
            Amplitude values over x in µm.

        Notes
        -----
        This methods uses the x and dist_x variables defined in :class:`Bpm`.
        """
        spot_size = fwhm / sqrt(2 * log(2))  # such as I=1/e^2 in intensity
        if spot_size != 0:
            field = np.exp(-(self.x / spot_size)**2)
            field = np.roll(field, int(round(offset_light / self.dist_x)))
        else:
            field = 0 * self.x  # Avoid division by zero error
        return field

    def squared_light(self, fwhm, offset_light=0):
        """
        Create a flat-top beam (squared).

        Parameters
        ----------
        fwhm : float
            Beam width in µm.
        offset_light : float, optional
            Light offset from center in µm. 0 by default.

        Returns
        -------
        field : array
            Amplitude values over x in µm.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_x, x.
        """
        field = np.zeros(self.nbr_x)

        for j in range(self.nbr_x):
            if self.x[j] >= -fwhm/2 and self.x[j] <= fwhm/2:
                field[j] = 1
            else:
                field[j] = 0

        field = np.roll(field, int(round(offset_light / self.dist_x)))
        return field

    def mode_determ(self, width, delta_no, mode):
        """
        Solve the transcendental equation tan=sqrt that give the modes
        allowed in a squared guide.

        Parameters
        ----------
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        delta_no : float
            Difference of refractive index between the core and the cladding.
        mode : int
            Number of the searched mode.

        Returns
        -------
        h_m : float
            Transverse propagation constant over x (µm).
        gamma_m : float
            Extinction coefficient over x (µm).
        beta_m : float
            Longitudinal constant of propagation over z (µm).

        Raises
        ------
        ValueError
            if no mode exists.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        lo, no, ko.
        """
        width = float(width)

        if width == 0:
            raise ValueError("no mode " + str(mode) + " existing")

        delta_no = float(delta_no.real)
        lim = self.lo/(2 * width * (self.no + delta_no)) - 1e-12
        theta_c = acos(self.no / (self.no + delta_no))  # Critical angle
        solu = np.linspace(
            mode*lim + 0.000001,
            (mode + 1) * lim,
            round(1 + (lim - 0.000001)/0.000001))
        lhs = np.tan(
            pi * width * (self.no + delta_no) / self.lo * solu
            - mode*pi/2)
        rhs = np.sqrt(
            0j  # to avoid sqrt error when complexe
            + (sin(theta_c) / solu)**2
            - 1)
        result = rhs - lhs  # 0 if left == right
        minimum = abs(result).min()  # return min value : where two equations~=
        i_min = int(np.where(abs(result) == minimum)[0])  # min value index

        if i_min == 0:
            raise ValueError("no mode " + str(mode) + " existing")

        sin_theta_m = solu[i_min]
        theta_m = asin(sin_theta_m)  # angle at which the mode propagate

        beta_m = self.ko * (self.no + delta_no) * cos(theta_m)
        h_m = sqrt((self.ko * (self.no + delta_no))**2 - beta_m**2)
        gamma_m = (self.no * self.ko
                   * np.sqrt((cos(theta_m) / cos(theta_c))**2 - 1))

        return [h_m, gamma_m, beta_m]

    def mode_light(self, width, delta_no, mode, offset_light=0):
        """
        Create light based on propagated mode inside a squared guide.

        Parameters
        ----------
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        delta_no : float
            Difference of refractive index between the core and the cladding.
        mode : int
            Number of the searched mode.
        offset_light : float, optional
            Light offset from center (µm). 0 by default.

        Returns
        -------
        field : array
            Amplitude values over x (µm).
        h_m : float
            Transverse propagation constant over x (µm).
        gamma_m : float
            Extinction coefficient over x (µm).
        beta_m : float
            Longitudinal constant of propagation over z (µm).

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_x, x and the :meth`mode_determ` method.
        """
        field = np.zeros(self.nbr_x)

        [h_m, gamma_m, beta_m] = self.mode_determ(width, delta_no, mode)

        if mode % 2 == 0:  # if even mode

            b_b = cos(h_m * width / 2)  # Continuity value where x=width/2

            for j in range(self.nbr_x):  # Compute light based on h,gamma,beta

                if abs(self.x[j]) <= width/2:  # in core
                    field[j] = cos(h_m * self.x[j])

                else:  # in cladding
                    field[j] = b_b * exp(-gamma_m * (
                        abs(self.x[j])
                        - width/2))
        else:  # if odd mode

            c_c = sin(h_m * width / 2)  # Continuity value where x=width/2

            for j in range(self.nbr_x):  # Compute light based on h,gamma,beta

                if abs(self.x[j]) <= width/2:  # in core
                    field[j] = sin(h_m * self.x[j])

                elif self.x[j] >= width/2:  # Right cladding
                    field[j] = c_c * exp(-gamma_m * (
                        self.x[j]
                        - width/2))

                else:  # Left cladding
                    field[j] = -c_c * exp(gamma_m * (
                        self.x[j]
                        + width/2))

        field = np.roll(field, int(round(offset_light / self.dist_x)))

        return [field, h_m, gamma_m, beta_m]

    def all_modes(self, width, delta_no, offset_light=0):
        """
        Compute all modes allowed by the guide and sum them into one field.

        Parameters
        ----------
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        delta_no : float
            Difference of refractive index between the core and the cladding.
        offset_light : float, optional
            Light offset from center in µm. 0 by default.

        Returns
        -------
        field : array
            Sum of all possibles fields in the guide.
        h : array, float
            Transverse propagation constant over x in µm of all modes.
        gamma : array, float
            Extinction coefficient over z in µm of all modes.
        beta : array, float
            Longitudinal constant of propagation over z in µm of all modes.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_x and the :meth`mode_light` method.
        """
        i = 0
        field = np.zeros(self.nbr_x)
        h = np.array([])
        gamma = np.array([])
        beta = np.array([])

        while True:
            try:
                [field_i, h_m, gamma_m, beta_m] = self.mode_light(
                    width, delta_no, i, offset_light)
                field = field + field_i
                h = np.append(h, h_m)
                gamma = np.append(gamma, gamma_m)
                beta = np.append(beta, beta_m)
                i += 1
            except ValueError:
                break

        return [field, h, gamma, beta]

    def check_modes(self, width, delta_no):
        """
        Return the last possible mode number.
        Could be merged with :meth:`all_modes` but would increase the needed
        time to compute just to display a number.

        Parameters
        ----------
        width : float
            Waveguide width (µm) at 1/e^2 intensity.
        delta_no : float
            Difference of refractive index between the core and the cladding.

        Returns
        -------
        m : int
            Number of the last possible mode for a squared guide.

        Notes
        -----
        This methods uses the :meth`mode_light` method defined in :class:`Bpm`.
        """
        i = 0

        while True:
            try:
                self.mode_light(width, delta_no, i)
                i += 1
            except ValueError:
                print("This guide can propagate up to the modes", i-1)
                return i-1

    def airy_light(self, lobe_size, airy_zero, offset_light=0):
        """
        Create an Airy beam using scipy.special.airy(x).

        Parameters
        ----------
        lobe_size : float
            Size of the first lobe (µm).
        airy_zero : int
            Cut the beam at the asked zero of the Airy function. n lobes will
            be displayed.
        offset_light : float, optional
            Light offset from center in µm. 0 by default.

        Returns
        -------
        field : array
            Amplitude values over x (µm).
        airy_zero : int
            Number of lobes. Corrected if higher than the window size.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_x, length_x, dist_x, x.
        """
        if lobe_size == 0 or airy_zero == 0:
            return [np.zeros(self.nbr_x), 0]

        lobe_size = -abs(lobe_size)

        # Position of the first zero and the asked one
        zero_pos = special.ai_zeros(airy_zero)[0]
        first_zero = zero_pos[0]
        last_zero = zero_pos[-1]

        # Positions/size of the wanted beam
        left = last_zero * lobe_size / first_zero
        right = 10 * lobe_size / first_zero  # Airy=1e-10 at x=10

        # Reduce the last zero number to stay in the window
        if -left > self.length_x:
            left = zero_pos * lobe_size / first_zero  # All possibles left posi
            airy_zero = np.where(-left <= self.length_x)[0]  # Higher index

            if airy_zero.size == 0:  # interface don't allow this situation
                print("The first lobe is bigger than the windows size")
                return [np.zeros(self.nbr_x), 0]

            else:  # take the higher lobe possible
                airy_zero = int(airy_zero[-1])

            last_zero = zero_pos[airy_zero]  # Value of the last lobe
            airy_zero += 1  # +1 to return the zero number

            left = last_zero * lobe_size / first_zero  # Corrected left positio

        # Number of points in the Airy window to match the full window
        nbr_point = int(round(abs((left - right) / self.dist_x)))

        # Airy window size
        x_airy = np.linspace(last_zero, 10, nbr_point)

        # Positions of the Airy and full window center
        center_airy = int(np.where(x_airy >= 0)[0][0])
        center = int(np.where(self.x >= 0)[0][0])

        # Airy field
        field = np.array(special.airy(x_airy)[0])

        # add last empty field to reach the windows size
        if self.x.size > field.size:
            field = np.append(field, np.zeros((self.x.size-field.size)))

        else:
            field.resize(self.x.size)  # Cut if exceed windows size

        # Recenter on 0
        field = np.roll(field, int(round(center - center_airy)))

        field = np.roll(field, int(round(offset_light / self.dist_x)))
        field /= np.max(field)  # Normalized

        return [field, airy_zero]

    def init_field(self, field, theta_ext, irrad):
        """
        Initialize phase, field and power variables.

        Parameters
        ----------
        field : array, array-like
            Amplitude values for each beams over x (µm) [beam,E] or E
        theta_ext : float
            Exterior inclinaison angle (°).
        irrad : array, array-like
            Irradiance for each beam (:math:`W/m^2`).

        Returns
        -------
        progress_pow : array
            Intensity values over x (µm).

        Notes
        -----
        This method creates the following variables within the class
        :class:`Bpm`:

        - epnc: Convertion factor used to set unit of the field and irradiance.
        - phase_mat: Free propagation in Fourier space over dz/2.
        - current_power: Intensity for z=0.
        - field: Field value with the unit.


        This methods uses the following variables defined in :class:`Bpm`:
        no, x, dist_x, nbr_x, nbr_z_disp.
        """
        assert theta_ext <= 28  # paraxial approximation limitation
        self.field = field.astype(complex)
        # see en.wiki: Gaussian_beam#Mathematical_form for intensity definition
        eta = 376.730313668  # Impedance of free space mu_0*c
        self.epnc = self.no / (2*eta)  # used to converte E into I
        #  unit(epnc)= W/V^2

        try:  # if multiple beams or one beam as [beam]
            _ = self.field.shape[1]  # Raise a IndexError if not
            nbr_light = self.field.shape[0]  # [beam1(x),beam2,beam3] -> 3
#            Eo = sqrt(irrad[i] / self.epnc)  # Peak value of the field (V/m).
            for i in range(nbr_light):
                self.field[i] *= sqrt(irrad[i] / self.epnc)

            self.field = np.sum(self.field, axis=0)  # merge all beam into one

        except IndexError:  # if only one beam and not in form [beam]
            self.field *= sqrt(irrad / self.epnc)

#        https://support.lumerical.com/hc/en-us/articles/
#        360034382894-Understanding-injection-angles-in-broadband-simulations
        theta = asin(sin(radians(theta_ext)) / self.no)  # angle in the guide
        ph = self.no * self.ko * sin(theta) * self.x  # k projection over x
        self.field *= np.exp(1j * ph)  # Initial phase due to angle

        nu_max = 1 / (2*self.dist_x)  # max frequency due to sampling
        # Spacial frequencies over x (1/µm)
        nu = np.linspace(-nu_max,
                         nu_max * (1 - 2/self.nbr_x),
                         self.nbr_x)
        intermed = self.no * cos(theta) / self.lo
        # Linear propagation phase
        fr = 2 * pi * nu**2 / (intermed + np.sqrt(
            intermed**2
            - nu**2
            + 0j))

        # Free space matrix
        self.phase_mat = fftshift(np.exp(-1j * self.dist_z / 2 * fr))

        # Initial irradiance
        self.current_power = self.epnc * abs2(self.field)

        self.progress_pow = np.zeros([self.nbr_z_disp, self.nbr_x])
        self.progress_pow[0, :] = np.array([self.current_power])

        return [self.progress_pow]

    def guide_position(self, peaks, guide, size):
        """
        Return the left and right position index over x of a given guide
        for each z.

        Parameters
        ----------
        peaks : array-like
            Central position of each guide [guide,z].
        guide : int
            Number of the guide.
        size : float
            Width (µm).

        Returns
        -------
        x_beg : array
            Left indices position of the selected guide over z.
        x_end : array
            Right indices position of the selected guide over z.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_z, x, length_x.
        """
        x_beg = np.array([None]*self.nbr_z)
        x_end = np.array([None]*self.nbr_z)

        if peaks.shape[0] != 0:

            for j in range(self.nbr_z):
                if peaks[guide, j] is None:
                    continue
                pos_beg = (peaks[guide, j] - size/2)  # Left position

                # If the position is out of boundery, change interval to
                # (-length_x/2, length_x)
                if pos_beg < self.x[0] or pos_beg > self.x[-1]:
                    pos_beg = pos_beg % self.length_x

                # If the pos_beg is between length_x/2 and length_x then change
                # interval to (-length_x/2, length_x/2)
                if pos_beg >= self.x[-1]:
                    pos_beg -= self.length_x

                # Search the closest index value for this position
                x_beg[j] = np.where(self.x >= pos_beg)[0][0]

                pos_end = (peaks[guide, j] + size/2)

                if pos_end < self.x[0] or pos_end > self.x[-1]:
                    pos_end = pos_end % self.length_x

                if pos_end >= self.x[-1]:
                    pos_end -= self.length_x

                x_end[j] = np.where(self.x >= pos_end)[0][0]
        return [x_beg, x_end]

    def power_guide(self, x_beg, x_end):
        """
        return the power over z in a given interval by integrating the beam
        irradiance.

        Parameters
        ----------
        x_beg : array
            Left indices position over z for a selected guide.
        x_end : array
            Right indices position over z for a selected guide.

        Returns
        -------
        P : array
            Normalized power in the guide over z.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        nbr_z_disp, progress_pow, pas.
        """
        P = np.zeros(self.nbr_z_disp)
        # explaination: power[0] is input so take dn[0] but dn[0] is for propag
        # from 0 to 1 unit so next power power[1] is also dn[0]
        for i in range(self.nbr_z_disp):
            if i == 0:
                index = 0
            elif i == self.nbr_z_disp-1:
                # -1 for beginning at 0 and -1 for final useless value
                index = len(x_beg)-2
            else:
                index = i*self.pas-1

            if x_beg[index] is None or x_end[index] is None:
                continue

            if x_beg[index] <= x_end[index]:
                P[i] = np.trapz(
                    self.progress_pow[i, x_beg[index]:x_end[index]],
                    dx=self.dist_x*1e-6)

            else:  # Take into account guides that crosses the window edges
                P[i] = np.trapz(
                    self.progress_pow[i, x_beg[index]:],
                    dx=self.dist_x*1e-6)
                P[i] += np.trapz(
                    self.progress_pow[i, :x_end[index]],
                    dx=self.dist_x*1e-6)

        P /= np.trapz(self.progress_pow[0], dx=self.dist_x*1e-6)
        return P  # f not normalized, unit: (W/m)

    def kerr_effect(self, dn, n2=None, chi3=None, kerr_loop=1,
                    variance_check=False, field_start=None,
                    dn_start=None, phase_mat=None):
        """
        Kerr effect: refractive index modulation by the light intensity.
        See: https://optiwave.com/optibpm-manuals/bpm-non-linear-bpm-algorithm/

        Parameters
        ----------
        dn : array
            Difference of refractive index as dn[z,x]. Can contain the losses
            throught the imaginary part.
        n2 : float, optional
            Nonlinear refractive index responsable for the optical Kerr effect
            in m^2/W. None by default.
        chi3 : float, optional
            Value of the third term of the electric susceptibility tensor
            in m^2/V^2. None by default.
        kerr_loop : int, optional
            Number of corrective loops for the Kerr effect. 1 by default.
        variance_check : bool, optional
            Check if the kerr effect converge fast enought. False by default.
        field_start : array, optional
            Field without kerr effect.
            If None were given, take the :meth:`main_compute` field.
        dn_start : array, optional
            Refractive index without kerr effect.
            If None were given, take the :meth:`main_compute` dn.
        phase_mat: array, optional
            Free propagation in Fourier space over dz/2.
            If None were given, take the :meth:`main_compute` phase_mat.

        Returns
        -------
        dn : array
            Refractive index with kerr effect.
        nl_mat : array
            refractive index modulation with kerr effect.
        field_x : array
            Field with the kerr effect at the self.i step.
        cur_pow : array
            Beam power with the kerr effect after the dz propagation.

        Notes
        -----
        This methods uses the following variables defined in :class:`Bpm`:
        i, epnc, no, ko, dist_z and the :meth:`variance` method.
        """
        assert n2 is None or chi3 is None
        # assert n2 is not None or chi3 is not None
        # Set the default value if none were given
        dn_start = dn[self.i, :] if dn_start is None else dn_start
        nl_mat = self.ko * self.dist_z * dn_start
        field_start = self.field if field_start is None else field_start
        phase_mat = self.phase_mat if phase_mat is None else phase_mat

        # Influence of the index modulation on the field
        field_x = field_start * np.exp(1j * nl_mat)

        # Linear propagation over dz/2
        field_x = ifft(phase_mat * fft(field_x))
        cur_pow = self.epnc * abs2(field_x)

        for _ in range(kerr_loop):
            prev_pow = cur_pow

            # influence of the beam intensity on the index modulation
            if n2 is not None:  # dn = dn1+dn2*I with I unit: W/m^2
                dn_kerr = dn_start + n2*prev_pow

            elif chi3 is not None:  # dn = dn1+ 3chi3/8*no*|E|^2
                dn_kerr = dn_start + (3*chi3)/(8*self.no)*(prev_pow/self.epnc)
            else:
                dn_kerr = dn_start  # identical to no kerr effect but slower

            nl_mat = self.ko * self.dist_z * dn_kerr

            # influence of the index modulation on the field
            field_x = field_start * np.exp(1j * nl_mat)

            # Linear propagation over dz/2
            field_x = ifft(phase_mat * fft(field_x))

            # power at pos z
            cur_pow = self.epnc * abs2(field_x)

        if variance_check:
            try:
                self.variance(prev_pow, cur_pow)  # Check if converge
            except ValueError as ex:
                print(ex)
                print("for the step i=", self.i)

            if np.max(dn_kerr) > self.no/10:
                print("Careful: index variation too high:",
                      "\t%.2f > %f/10" % (np.max(dn_kerr), self.no), sep="\n")

        return [dn_kerr, nl_mat, field_x, cur_pow]

    def variance(self, initial, final):
        """
        This function alerts the user when the kerr effect don't converge fast
        enough.
        Raise a ValueError when the power standard deviation exceed
        :math:`10^{-7}`.

        Parameters
        ----------
        initial : array
            Power of the kerr effect looped n-1 time.
        final : array
            Power of the kerr effect looped n time.

        Raises
        ------
        ValueError
            when the power standard deviation exceed :math:`10^{-7}`.
        """
        finish_sum = np.sum(final)
        nl_control_amp = 1/finish_sum * np.sqrt(np.abs(
            np.sum(np.subtract(final, initial)**2)
            - np.sum(np.subtract(final, initial))**2))

        if nl_control_amp > 1e-7:
            message = "Warning: solution doesn't converge " + \
                "for a deviation of " + str(nl_control_amp)
            raise ValueError(message)

    def bpm_compute(self, dn, n2=None, chi3=None, kerr_loop=1,
                    variance_check=False):
        """
        Compute BPM principle : free_propag over dz/2, index modulation,
        free_propag over dz/2.

        Parameters
        ----------
        n2 : float, optional
            Nonlinear refractive index responsable for the optical Kerr effect
            in m^2/W. None by default.
        chi3 : float, optional
            Value of the third term of the electric susceptibility tensor
            in m^2/V^2. None by default.
        kerr : bool, optional
            Activate the kerr effect. False by default.
        kerr_loop : int, optional
            Number of corrective loops for the Kerr effect. 1 by default.
        variance_check : bool
            Check if the kerr effect converge fast enought. False by default.

        Returns
        -------
        current_power : array
            Power after the propagation over dz.

        Notes
        -----
        This method uses the :class:`Bpm` class variables:
        nbr_lost, i, field, dist_z, dn, nl_mat, phase_mat, epnc,
        :meth:`kerr_effect`.

        This method change the values of:
        field, dn, nl_mat, current_power.
        """
        # Linear propagation over dz/2
        self.field = ifft(self.phase_mat * fft(self.field))

        if n2 is not None or chi3 is not None:
            [dn[self.i, :], self.nl_mat[self.i, :],
             self.field, self.current_power] = self.kerr_effect(
                 dn, n2=n2, chi3=chi3, kerr_loop=kerr_loop,
                 variance_check=variance_check)
        else:
            # Influence of the index modulation on the field (contain losses)
            self.field *= np.exp(1j * self.nl_mat[self.i, :])

            # Linear propagation over dz/2
            self.field = ifft(self.phase_mat * fft(self.field))

            # power at z
            self.current_power = self.epnc * abs2(self.field)

        # useless but act as a reminder for what the the method does
        return self.current_power

    def main_compute(self, dn, n2=None, chi3=None, kerr_loop=1,
                     variance_check=False, disp_progress=True):
        """
        main method used to compute propagation.

        Parameters
        ----------
        n2 : float, optional
            Nonlinear refractive index responsable for the optical Kerr effect
            in m^2/W. None by default.
        chi3 : float, optional
            Value of the third term of the electric susceptibility tensor
            in m^2/V^2. None by default.
        kerr : bool, optional
            Activate the kerr effect. False by default.
        kerr_loop : int, optional
            Number of corrective loop for the Kerr effect. 1 by default.
        variance_check : bool, optional
            Check if the kerr effect converge fast enought. False by default.
        alpha : float, optional
            Absorption per µm. 0 by default
        lost_beg : array-like, optional
            Left indices position of the selected waveguide over z.
            None by default.
        lost_end : array-like, optional
            Right indices position of the selected waveguide over z.
            None by default.

        Returns
        -------
        progress_pow : array
            Intensity values (:math:`W/m^2`) over x (µm) and z (µm).

        Notes
        -----

        This method creates the following variables within the class
        :class:`Bpm`:
        nl_mat: Refractive index modulation.

        This method uses the :class:`Bpm` class variables:
        phase_mat, field, i, nbr_z, pas, current_power, dist_z, length_z,
        nbr_lost, dn, nl_mat, epnc and uses the :meth:`bpm_compute`,
        :meth:`kerr_effect`.

        This method change the values of the :class:`Bpm` class variables:
        field and if kerr, dn and nl_mat.
        """
        # Refractive index modulation
        self.nl_mat = self.ko * self.dist_z * dn

        index = 0
        self.i = 0
        #  from i=0 to i=final-1 because don't use last dn
        for i in range(self.nbr_z-1):
            self.i = i
            # Compute non-linear and linear propagation for every z
            self.bpm_compute(dn, n2=n2, chi3=chi3, kerr_loop=kerr_loop,
                             variance_check=variance_check)

            # Display condition: if i+1 is a multiple of pas: i+1 % pas = 0
            # = False, so must use if not to have True
            # last condition to have last point if not a multiple of pas
            if not (self.i + 1) % self.pas or self.i+1 == self.nbr_z-1:
                index += 1
                self.progress_pow[index, :] = np.array([self.current_power])

                if disp_progress:
                    current = (self.i+1)*self.dist_z/1e3
                    print(current, "/", self.length_z/1e3, 'mm')

        return [self.progress_pow]


def example_bpm():
    """
    Version allowing to compute BPM without the user interface.
    Used for quick test.
    """
    lo = 1.5
    width = 6
    no = 2.14
    # losses = 0.8/1e3
    # no_imag = 1.9e-4
    # no_imag = losses/(2*pi/lo)
    delta_no = 0.001  # + 1j*no_imag
    length_z = 2000
    dist_z = 1
    nbr_z_disp = 200
    dist_x = 0.1
    length_x = 500

    bpm = Bpm(no, lo,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

#    shape = bpm.squared_guide(width)
    shape = bpm.gauss_guide(width, 4)

    nbr_p = 3
    p = 13
    offset_guide = 0

    [peaks, dn] = bpm.create_guides(
        shape, delta_no, nbr_p, p, offset_guide=offset_guide)
#    curve = 40 * 1E-8
#    half_delay = 1000
#    distance_factor = 1.2
#    [peaks, dn] = bpm.create_curved_guides(shape, width, delta_no,
#                                           curve, half_delay,
#                                           distance_factor,
#                                           offset_guide=offset_guide)

    z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)
    xv, zv = np.meshgrid(x, z_disp)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)

#    plt.figure()
#    for i in range(nbr_z_disp+1):
#        plt.plot(x,dn[i,:])

    plt.figure()
    plt.title("Refractive index over (x,z)")
    plt.xlabel('x (µm)')
    plt.ylabel('z (mm)')
    plt.pcolormesh(xv,
                   zv,
                   dn[dn_disp].real,
                   cmap='gray')

    # ax1 = plt.subplot()
    # ax1.set_title("Losses map")
    # ax1.set_xlabel('x (µm)')
    # ax1.set_ylabel('z (mm)')
    # ax1.pcolormesh(xv, zv, dn[dn_disp].imag)

    fwhm = 8
    offset_light = peaks[0, 0]  # If guide exists
    offset_light = 0  # Else

    nbr_light = 1
    field = np.array([np.zeros(nbr_x)] * nbr_light)
    for i in range(nbr_light):
        field_i = bpm.gauss_light(fwhm, offset_light=offset_light)

    #    field_i = bpm.squared_light(fwhm, offset_light=offset_light)

#            [field_i, h, gamma, beta] = bpm.all_modes(
#                width, delta_no
#                offset_light=offset_light)

#        mode = 0
#            [field_i, h, gamma, beta] = bpm.mode_light(
#                width, delta_no,
#                mode, offset_light=offset_light)

        field[i] = field_i

    irrad = [1E13]*nbr_light
    theta_ext = 0
    [progress_pow] = bpm.init_field(field, theta_ext, irrad)

    def _show_plot(pow_index):
        plt.figure()
        ax1 = plt.subplot(111)
        if pow_index == 0:
            ax1.set_title("Light injection into a guide")
        else:
            ax1.set_title("Light at the end of guides")
        ax1.set_xlabel('x (µm)')
        ax2 = ax1.twinx()
        for tl in ax1.get_yticklabels():
            tl.set_color('k')
        for tl in ax2.get_yticklabels():
            tl.set_color('#1f77b4')
        ax1.set_ylabel(r'$\Delta_n$')
        ax2.set_ylabel('Irradiance ($GW.cm^{-2}$)')

        if nbr_p != 0 and p != 0:
            ax1.set_xlim(-nbr_p*p, nbr_p*p)
            verts = [(x[0], 0),
                     *zip(x, dn[pow_index, :].real),
                     (x[-1], 0)]
            poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
            ax1.add_patch(poly)
        ax1.set_ylim(0,
                     max(dn[0, :].real)*1.1 + 1E-20
                     )
        if max(progress_pow[0]) != 0:
            ax2.set_ylim(0,  1.1e-13*max(progress_pow[0]))

        ax1.plot(x, dn[pow_index].real, 'k')
        ax2.plot(x, 1e-13*progress_pow[pow_index], '#1f77b4')
        plt.show()

    pow_index = 0
    print("May need to close the graph to continue.")
    _show_plot(pow_index)

    kerr_loop = 5
    variance_check = False
#    n2 = 1e-16
#    chi3 = 10 * 1E-22
    n2 = None
    chi3 = None

    kerr = n2 is not None or chi3 is not None

    estimation = round(
        8.8 / 5e7 * nbr_z * nbr_x  # without kerr
        * (1 + 0.72*float(kerr)*(kerr_loop))  # with kerr
        + 3.8e-8*nbr_z*nbr_x*float(variance_check),  # control
        1)

    print("Time estimate:", str(estimation))

    debut = time.process_time()
    [progress_pow] = bpm.main_compute(
        dn,
        n2=n2, chi3=chi3, kerr_loop=kerr_loop,
        variance_check=variance_check, disp_progress=False)
    fin = time.process_time()
    print('time elapsed:', fin-debut)

    plt.figure()
    ax1 = plt.subplot(111)
    ax1.set_title("Light propagation into guides")
    ax1.set_xlabel('x (µm)')
    ax1.set_ylabel('z (mm)')
    if nbr_p != 0 and p != 0:
        ax1.set_xlim(-nbr_p*p, nbr_p*p)
    ax1.pcolormesh(xv, zv, 1e-13*progress_pow)

    pow_index = -2
    print("May need to close the graphs to continue.")
    _show_plot(pow_index)
    print("Finished")


if __name__ == "__main__":
    print("version without the user interface, note that user_interface.py")
    print("calls the Bpm class and performs the same calculations")

    choice = input("Start ?: ")

    if not choice != "yes":
        example_bpm()
