from beampy.bpm import Bpm
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift
from math import pi, ceil, radians, sqrt, log, sin, cos, acos, asin, exp


def example_gaussian_beam():
    """ Display a Gaussian beam with the fwhm definition."""
    fwhm = 6
    bpm = Bpm(1, 1, 1, 1, 1, 1, 1, 1)
    bpm.x = np.linspace(-15, 9.1, 500)
    x = bpm.x
    plt.figure("Beam")
    plt.title("Example for the gaussian beam")
    plt.plot(x, bpm.gauss_light(fwhm, 0), label='field')
    plt.plot(x, (bpm.gauss_light(fwhm, 0))**2, label='intensity')
    plt.plot(x, [1/2]*x.size, '-.', label='1/2')
    plt.plot([fwhm/2]*x.size, np.linspace(0, 1, x.size), '--', label='fwhm/2')
    plt.plot(x, [np.exp(-1)]*x.size, '-.', label='1/e')
    plt.plot(x, [np.exp(-2)]*x.size, '-.', label='1/$e^2$')
    plt.plot([fwhm / np.sqrt(2 * np.log(2))]*x.size, np.linspace(0, 1, x.size),
             '--', label='$w_0$')
    plt.legend()
    plt.show()


def example_guides_x():
    """Display a Gaussian guide, two super-Gaussian guides and a flat-top guide
    to illustrate the width definition."""
    width = 6
    bpm = Bpm(width, 1, 1, 1, 1, 1, 1, 1)
    bpm.x = np.linspace(-15, 9.1, 500)
    x = bpm.x
    plt.figure("guide_x")
    plt.title("Example of different guides")
    plt.plot(x, bpm.gauss_guide(1)(x), label='Gaussian')
    plt.plot(x, bpm.gauss_guide(4)(x), label='super-Gaussian P=4')
    plt.plot(x, bpm.gauss_guide(10)(x), label='super-Gaussian P=10')
    plt.plot(x, bpm.squared_guide()(x), label='Flat-top')
    plt.plot([width/2]*x.size, np.linspace(0, 1, x.size), '--',
             label='width/2')
    plt.plot(x, [np.exp(-1)]*x.size, '-.', label='1/e')
    plt.legend()
    plt.show()


def example_guides_z():
    """Display an array of guides and the curved guides system."""
    width = 6
    bpm = Bpm(width, 2, 0.1, 10000, 1, 200, 100, 0.1)
    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()
    shape = bpm.gauss_guide(4)
    [peaks, dn] = bpm.create_guides(shape, 5, 10)
    z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)
    xv, zv = np.meshgrid(x, z_disp)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)
    plt.figure("guide_z_array")
    plt.title("Example for the array of guides")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()
    [peaks, dn] = bpm.create_curved_guides(shape, 40*1e-8, 1000, 1.2)
    plt.figure("guide_z_curved")
    plt.title("Example for the curved guides")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()


def example_free_propag():
    """Show the free propagation of a beam (no refractive index variation)
    and confirm that Beampy return the correct waist value"""
    width = 0
    no = 1
    delta_no = 0
    length_z = 10000
    dist_z = 10000
    nbr_z_disp = 1
    dist_x = 0.01
    length_x = 1000

    bpm = Bpm(width, no, delta_no,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape = bpm.squared_guide()

    nbr_p = 0
    p = 0

    [peaks, dn] = bpm.create_guides(shape, nbr_p, p)

    fwhm = 20
    lo = 1.5

    field = bpm.gauss_light(fwhm)
    irrad = 1E13
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad, lo)

    [progress_pow] = bpm.main_compute()

    intensity = progress_pow[-1]

    fwhm2 = np.where(
            intensity >= (max(intensity)/2)
    )[0][0]

    fwhm_final = abs(2 * x[fwhm2])

    w0_final = fwhm_final / np.sqrt(2 * np.log(2))
    print("Beampy:", w0_final)

    w0 = fwhm / np.sqrt(2 * np.log(2))
    z0 = np.pi * w0**2 / lo
    w = w0 * np.sqrt(1 + (length_z / z0)**2)

    print("Theory:", w)
    print("relative difference:", abs(w - w0_final)/w*100, "%")


def example_stability():
    """Show the possible BPM approximations for implementing a refractive
    index variation"""

    width = 6
    no = 2.14
    delta_no = 0.0014
    length_z = 200
    dist_z = 10
    nbr_z_disp = 1
    dist_x = 0.1
    length_x = 1000

    bpm = Bpm(width, no, delta_no,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape = bpm.squared_guide()

    nbr_p = 1
    p = 0

    [peaks, dn] = bpm.create_guides(shape, nbr_p, p)

    fwhm = 6
    lo = 1.5

    field = bpm.gauss_light(fwhm)
    irrad = 1
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad, lo)

    nbr_step = 10

    # Need to overwrite those variables due to changes
    theta = asin(sin(radians(theta_ext)) / no)  # angle in the guide
    nu_max = 1 / (2 * dist_x)  # max frequency due to sampling
    # Spacial frequencies over x (1/µm)
    nu = np.linspace(-nu_max,
                     nu_max * (1 - 2/nbr_x),
                     nbr_x)
    intermed = no * cos(theta) / lo
    fr = -2 * pi * nu**2 / (intermed + np.sqrt(
        intermed**2
        - nu**2
        + 0j))

    bpm.phase_mat = fftshift(np.exp(1j * dist_z * fr))
    bpm.phase_mat_demi = fftshift(np.exp(1j * dist_z / 2 * fr))
    # End overwrite

    field = bpm.field
    for i in range(nbr_step):
        field = ifft(bpm.phase_mat * fft(field))
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
    test_1 = field

    field = bpm.field
    for i in range(nbr_step):
        field = ifft(bpm.phase_mat_demi * fft(field))
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
        field = ifft(bpm.phase_mat_demi * fft(field))
    test_2 = field

    field = bpm.field
    for i in range(nbr_step):
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
        field = ifft(bpm.phase_mat * fft(field))
    test_3 = field

    plt.figure("field real")
    plt.title("Impact of the free propagation order")
    plt.xlim(-20, 20)
    plt.ylim(-1, 20)
    plt.plot(x, test_1.real, label='first: dz+lens')
    plt.plot(x, test_2.real, label='middle: dz/2+lens+dz/2')
    plt.plot(x, test_3.real, label='last: lens+dz')
    plt.legend()
    plt.show()
    plt.figure("field imag")
    plt.title("Impact of the free propagation order")
    plt.xlim(-30, 30)
    plt.plot(x, test_1.imag, label='first: dz+lens')
    plt.plot(x, test_2.imag, label='middle: dz/2+lens+dz/2')
    plt.plot(x, test_3.imag, label='last: lens+dz')
    plt.legend()
    plt.show()

    field = bpm.field
    field = ifft(bpm.phase_mat_demi * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat_demi * fft(field))
    test_4 = field

    field = bpm.field
    field = ifft(bpm.phase_mat_demi * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat_demi * fft(field))
    field = ifft(bpm.phase_mat_demi * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat_demi * fft(field))
    test_5 = field

    plt.figure("field real 2")
    plt.title("Same algorithme optimized")
    plt.xlim(-20, 20)
    plt.ylim(-1, 20)
    plt.plot(x, test_4.real, label='dz/2+lens+dz+lens+dz/2')
    plt.plot(x, test_5.real, label='(dz/2+lens+dz/2)*2')

    plt.legend()
    plt.show()
    plt.figure("field imag 2")
    plt.title("Same algorithme optimized")
    plt.xlim(-30, 30)
    plt.plot(x, test_4.imag, label='dz/2+lens+dz+lens+dz/2')
    plt.plot(x, test_5.imag, label='(dz/2+lens+dz/2)*2')
    plt.legend()
    plt.show()

    field = bpm.field
    field = ifft(bpm.phase_mat_demi * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat * fft(field))
    test_6 = field

    field = bpm.field
    field = ifft(bpm.phase_mat_demi * fft(field))
    field *= np.exp(1j * bpm.nl_mat[0, :])
    field = ifft(bpm.phase_mat_demi * fft(field))
    test_7 = field

    plt.figure("field real 3")
    plt.title("Approximation if uses loop over lens+dz")
    plt.xlim(-20, 20)
    plt.ylim(-1, 20)
    plt.plot(x, test_6.real, label='dz/2+lens+dz')
    plt.plot(x, test_7.real, label='dz/2+lens+dz/2')

    plt.legend()
    plt.show()
    plt.figure("field imag 3")
    plt.title("Approximation if uses loop over lens+dz")
    plt.xlim(-30, 30)
    plt.plot(x, test_4.imag, label='dz/2+lens+dz')
    plt.plot(x, test_5.imag, label='dz/2+lens+dz/2')
    plt.legend()
    plt.show()


def example_kerr():
    """More test than example.
    Show the different approximation possible for the BPM implementation of the
    Kerr effect."""
    width = 6
    no = 1
    delta_no = 0.0014
    length_z = 1000
    dist_z = 0.5
    nbr_z_disp = 1
    dist_x = 0.1
    length_x = 300.8

    bpm = Bpm(width, no, delta_no,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape = bpm.gauss_guide(4)

    nbr_p = 0
    p = 0

    [peaks, dn] = bpm.create_guides(shape, nbr_p, p)

    fwhm = 6
    lo = 1.5

    field = bpm.gauss_light(fwhm)
    irrad = 20000e13  # if too high, see big difference between method
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad, lo)

    # Need to overwrite those variables due to changes
    theta = asin(sin(radians(theta_ext)) / no)  # angle in the guide
    nu_max = 1 / (2 * dist_x)  # max frequency due to sampling
    # Spacial frequencies over x (1/µm)
    nu = np.linspace(-nu_max,
                     nu_max * (1 - 2/nbr_x),
                     nbr_x)
    intermed = no * cos(theta) / lo
    fr = -2 * pi * nu**2 / (intermed + np.sqrt(
        intermed**2
        - nu**2
        + 0j))

    bpm.phase_mat = fftshift(np.exp(1j * dist_z * fr))
    bpm.phase_mat_demi = fftshift(np.exp(1j * dist_z / 2 * fr))
    # End overwrite

    kerr_loop = 3
    variance_check = False
    chi3 = 10 * 1E-20

    nbr_step = 2000  # max length_z / dist_z

    print("\n dz/2+lens+dz/2")
    field = bpm.field
    for i in range(nbr_step):
        print("step", i)
#        plt.figure(num='Reference without kerr')
#        ax1 = plt.subplot(211)
#        ax2 = plt.subplot(212)
#        ax1.set_title("real: no kerr")
#        ax2.set_title("imag: no kerr")
#        plt.xlim(-1, 1)
#        plt.ylim(5.85e17, 6.05e17)

        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        # Influence of the index modulation on the field
        field = field * np.exp(1j * bpm.nl_mat[nbr_step, :])  # No changes if
        # no initial guide (exp=1)

        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        cur_pow = bpm.epnc * (field * field.conjugate()).real
#        ax1.plot(x, field_x.real, label='no kerr')
#        ax2.plot(x, field_x.imag, label='no kerr')

    field_ref = field
    cur_ref = cur_pow
#    ax1.legend(loc="upper right")
#    ax2.legend(loc="upper right")
#    plt.show()

    print("\n dz+kerr")
    field = bpm.field
    for i in range(nbr_step):
        print("step", i)
#        plt.figure(num='Impact of the kerr effect for dz+kerr')
#        ax1 = plt.subplot(211)
#        ax2 = plt.subplot(212)
#        ax1.set_title("real: dz+kerr")
#        ax2.set_title("imag: dz+kerr")
#        plt.xlim(-1, 1)
#        plt.ylim(5.85e17, 6.05e17)

        # Linear propagation over dz
        field = ifft(bpm.phase_mat * fft(field))

        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])  # No changes if
        # no initial guide (exp=1)

        cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

#        ax1.plot(x, field_x.real, label='no kerr')
#        ax2.plot(x, field_x.imag, label='no kerr')

        if kerr_loop != 0:
            for k in range(1):
                prev_pow = cur_pow
                # influence of the beam intensity on the index modulation
                dn = bpm.dn[nbr_step, :] + (3 * chi3 / 8 / no * prev_pow)
                nl_mat = bpm.ko * bpm.dist_z * dn

                # influence of the index modulation on the field
                field_x = field * np.exp(1j * nl_mat)  # No changes for the pow

                # power at z
                cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

    #            ax1.plot(x, field_x.real, label='loop:'+str(k+1))
    #            ax2.plot(x, field_x.imag, label='loop:'+str(k+1))

    #        print(max(dn))
    #        if variance_check:
    #            try:
    #                bpm.variance(prev_pow, cur_pow)  # Check if converge
    #                    print(bpm.nl_control_amp)
    #            except ValueError as ex:
    #                    print(ex)
    #                print("Warning", bpm.nl_control_amp)

        field = field_x

    field_1 = field
    cur_1 = cur_pow
    dn_1 = dn
#    ax1.legend(loc="upper right")
#    ax2.legend(loc="upper right")
#    plt.show()

    print("\n dz/2+kerr+dz/2")
    field = bpm.field
    for i in range(nbr_step):
        print("step", i)
#        plt.figure(num="intensity with kerr dz/2+kerr+dz/2")
#        ax1 = plt.subplot(211)
#        ax2 = plt.subplot(212)
#        ax1.set_title("real: dz/2+kerr+dz/2")
#        ax2.set_title("imag: dz/2+kerr+dz/2")
#        plt.xlim(-1, 1)
#        plt.ylim(5.85e17, 6.05e17)

        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])  # No changes if
        # no initial guide (exp=1)

        # Linear propagation over dz/2
        field_x = ifft(bpm.phase_mat_demi * fft(field_x))

        cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

#        ax1.plot(x, field_x.real, label='no kerr')
#        ax2.plot(x, field_x.imag, label='no kerr')

        for k in range(kerr_loop):
            prev_pow = cur_pow
            # influence of the beam intensity on the index modulation
            dn = bpm.dn[nbr_step, :] + (3 * chi3 / 8 / no * prev_pow)
            nl_mat = bpm.ko * bpm.dist_z * dn

            # influence of the index modulation on the field
            field_x = field * np.exp(1j * nl_mat)
            # Linear propagation over dz/2
            field_x = ifft(bpm.phase_mat_demi * fft(field_x))
            # power at z
            cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

#            ax1.plot(x, field_x.real, label='loop:'+str(k+1))
#            ax2.plot(x, field_x.imag, label='loop:'+str(k+1))

#        print(max(dn))
        if variance_check:
            try:
                bpm.variance(prev_pow, cur_pow)  # Check if converge
#                    print(bpm.nl_control_amp)
            except ValueError:
                print("Warning", bpm.nl_control_amp)

        field = field_x

    field_2 = field
    cur_2 = cur_pow
    dn_2 = dn
#    ax1.legend(loc="upper right")
#    ax2.legend(loc="upper right")
#    plt.show()

    print("\n kerr+dz")
    field = bpm.field
    for i in range(nbr_step):
        print("step", i)

#        plt.figure(num="intensity with kerr kerr+dz")
#        ax1 = plt.subplot(211)
#        ax2 = plt.subplot(212)
#        ax1.set_title("real: kerr+dz")
#        ax2.set_title("imag: kerr+dz")
#        plt.xlim(-1, 1)
#        plt.ylim(5.85e17, 6.05e17)

        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])  # No changes if
        # no initial guide (exp=1)

        # Linear propagation over dz
        field_x = ifft(bpm.phase_mat * fft(field_x))

        cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

#        ax1.plot(x, field_x.real, label='no kerr')
#        ax2.plot(x, field_x.imag, label='no kerr')

        for k in range(kerr_loop):
            prev_pow = cur_pow
            # influence of the beam intensity on the index modulation
            dn = bpm.dn[nbr_step, :] + (3 * chi3 / 8 / no * prev_pow)
            nl_mat = bpm.ko * bpm.dist_z * dn

            # influence of the index modulation on the field
            field_x = field * np.exp(1j * nl_mat)
            # Linear propagation over dz
            field_x = ifft(bpm.phase_mat * fft(field_x))
            # power at z
            cur_pow = bpm.epnc * (field_x * field_x.conjugate()).real

#            ax1.plot(x, field_x.real, label='loop:'+str(k+1))
#            ax2.plot(x, field_x.imag, label='loop:'+str(k+1))

#        print(max(dn))
        if variance_check:
            try:
                bpm.variance(prev_pow, cur_pow)  # Check if converge
#                    print(bpm.nl_control_amp)
            except ValueError:
                print("Warning", bpm.nl_control_amp)

        field = field_x

    field_3 = field
    cur_3 = cur_pow
    dn_3 = dn
#    ax1.legend(loc="upper right")
#    plt.show()

    plt.figure(num="Impact order kerr")

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.set_title("phase: comparison")
    ax2.set_title("power: comparison")

    ax1.set_xlim(-200, 200)
    ax2.set_xlim(-200, 200)

    ax1.plot(x, np.angle(field_ref), label="no kerr")
    ax1.plot(x, np.angle(field_1), label="dz+kerr")
    ax1.plot(x, np.angle(field_2), label="dz/2+kerr+dz/2")
    ax1.plot(x, np.angle(field_3), label="kerr+dz")

    ax2.plot(x, cur_ref, label="no kerr")
    ax2.plot(x, cur_1, label="dz+kerr")
    ax2.plot(x, cur_2, label="dz/2+kerr+dz/2")
    ax2.plot(x, cur_3, label="kerr+dz")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    plt.show()

    dn_ref = bpm.dn[nbr_step, :]

    plt.figure(num="Impact on dn order kerr")

    ax1 = plt.subplot(111)

    ax1.set_title("dn: comparison")

    ax1.set_xlim(-200, 200)

    ax1.plot(x, dn_ref, label="no kerr")
    ax1.plot(x, dn_1, label="dz+kerr")
    ax1.plot(x, dn_2, label="dz/2+kerr+dz/2")
    ax1.plot(x, dn_3, label="kerr+dz")

    ax1.legend(loc="upper right")

    plt.show()
