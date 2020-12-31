from math import pi, ceil, radians, sqrt, log, sin, cos, acos, asin, exp

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from numpy.fft import fft, ifft, fftshift

from beampy.bpm import Bpm, abs2


def gaussian_beam(fwhm=10):
    """Display a Gaussian beam at the given fwhm.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum (for intensity not amplitude) (µm).
    """
    w = fwhm / np.sqrt(2 * np.log(2))
    bpm = Bpm(1, 1, 1, 1, 1, 1, 1)
    bpm.x = np.linspace(-1.5*fwhm, 1.5*fwhm, 500)
    x = bpm.x
    plt.figure("Beam")
    plt.title("Width definition of a gaussian beam with FWHM=%s µm" % fwhm)
    plt.xlabel("x (µm)")
    plt.plot(x, bpm.gauss_light(fwhm, 0), label='field')
    plt.plot(x, (bpm.gauss_light(fwhm, 0))**2, label='intensity')
    plt.plot(x, [1/2]*x.size, '-.', label='1/2')
    plt.plot([fwhm/2]*x.size, np.linspace(0, 1, x.size),
             '--', label='fwhm/2')
    plt.plot(x, [np.exp(-1)]*x.size, '-.', label='1/e')
    plt.plot(x, [np.exp(-2)]*x.size, '-.', label='1/$e^2$')
    plt.plot([w]*x.size, np.linspace(0, 1, x.size),
             '--', label='$w_0$=%.2f' % w)
    plt.legend()
    plt.show()
#    plt.savefig("def_gauss_beam.png", bbox="tight", dpi=250)


def guides_x():
    """Display a Gaussian guide, two super-Gaussian guides and a flat-top guide
    to illustrate the width definition."""
    width = 10
    bpm = Bpm(1, 1, 1, 1, 1, 1, 1)
    bpm.x = np.linspace(-15, 9.1, 500)
    x = bpm.x
    plt.figure("guide_x")
    plt.title("Different waveguides shape available")
    plt.plot(x, bpm.gauss_guide(width, 1)(x), label='Gaussian')
    plt.plot(x, bpm.gauss_guide(width, 4)(x), label='super-Gaussian P=4')
    plt.plot(x, bpm.gauss_guide(width, 10)(x), label='super-Gaussian P=10')
    plt.plot(x, bpm.squared_guide(width)(x), label='Flat-top')
    plt.plot([width/2]*x.size, np.linspace(0, 1, x.size), '--',
             label='width/2')
    plt.plot(x, [np.exp(-1)]*x.size, '-.', label='1/e')
    plt.legend()
    plt.show()
#    plt.savefig("def_gauss_guide.png", bbox="tight", dpi=250)


def guides_z():
    """Display an array of guides and the curved guides system."""
    width = 6
    bpm = Bpm(2, 1.55, 10000, 1, 200, 100, 0.1)
    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()
    shape = bpm.gauss_guide(width, 4)
    [peaks, dn] = bpm.create_guides(shape, 0.01, 5, 10)
    z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)
    xv, zv = np.meshgrid(x, z_disp)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)
    plt.figure("Waveguide array")
    plt.title("Waveguides array example")
    plt.xlabel("x (µm)")
    plt.ylabel("z (mm)")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()
#    plt.savefig("waveguide_array.png", bbox="tight", dpi=250)

    [peaks, dn] = bpm.create_curved_guides(
            shape, width, 0.01, 40*1e-8, 1000, 1.2)
    plt.figure("Curved wavewaveguides")
    plt.title("Curved waveguides example")
    plt.xlabel("x (µm)")
    plt.ylabel("z (mm)")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()
#    plt.savefig("waveguide_curved.png", bbox="tight", dpi=250)

    shape = bpm.gauss_guide(width, 4)
    dn = bpm.create_guides(shape, 0.01, 1, 10, z=[0, length_z/2])[1]
    dn2 = bpm.create_guides(shape, 0.01, 2, 10,
                            z=[length_z/2, length_z])[1]
    dn3 = bpm.create_guides(shape, 0.01, 2, 30,
                            z=[length_z/4, 3*length_z/4])[1]
    dn4 = bpm.create_guides(shape, 0.01, 1, 0, offset_guide=40)[1]
    dn = np.add(dn, dn2)
    dn = np.add(dn, dn3)
    dn = np.add(dn, dn4)
    z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)
    xv, zv = np.meshgrid(x, z_disp)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)
    plt.figure("Arbitrary waveguides example")
    plt.title("Arbitrary waveguides example with discontinuity")
    plt.xlabel("x (µm)")
    plt.ylabel("z (mm)")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()
#    plt.savefig("waveguide_arbitrary.png", bbox="tight", dpi=250)


def free_propag(dist_x=0.1, length_x=1000, length_z=10000, no=1, lo=1):
    """Show the free propagation of a beam and compare Beampy results with
    the theorical values.

    Parameters
    ----------
    dist_x : float
        Step over x (µm).
    length_x : float
        Size of the compute window over x (µm).
    length_z : float
        Size of the compute window over z (µm).
    no : float
        Refractive index of the cladding
    lo : float
        Wavelength of the beam in vaccum (µm).
    """

    dist_z = length_z
    nbr_z_disp = 1

    bpm = Bpm(no, lo,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    dn = np.zeros((nbr_z, nbr_x))

    fwhm = 20

    field = bpm.gauss_light(fwhm)
    irrad = 1E13
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad)

    [progress_pow] = bpm.main_compute(dn)

    intensity = progress_pow[-1]

    index = np.where(intensity >= (np.max(intensity)/2))[0][0]
    fwhm_final = abs(2 * x[index])

    w0_final = fwhm_final / np.sqrt(2 * np.log(2))
    print("Beampy: width w = %f µm" % w0_final)

    w0 = fwhm / np.sqrt(2 * np.log(2))
    z0 = np.pi * no * w0**2 / lo
    w = w0 * np.sqrt(1 + (length_z / z0)**2)
    print("Theory: width w = %f µm" % w)

    diff = abs(w - w0_final)/w*100
    print("Relative difference: %f" % diff, "%")

    if diff > 1:
        print("Check the dist_x or length_x parameters")

    irrad_theo = irrad*w0/w  # in 2D: irrad*(w0/w)**2

    print("Beampy: irradiance I =", np.max(intensity))
    print("Theory: irradiance I =", irrad_theo)

    diff = abs(np.max(intensity) - irrad_theo)/irrad_theo*100
    print("Relative difference: %f" % diff, "%")

    if diff > 1:
        print("Check the dist_x or length_x parameters")

    fwhm_theo = w * np.sqrt(2 * np.log(2))

    profil_theo = bpm.gauss_light(fwhm_theo)
    intensity_theo = irrad_theo * abs2(profil_theo)

    plt.figure()
    plt.title("Beam profil after %.0f mm of free propagation" % (length_z/1e3))
    plt.xlabel("x (µm)")
    plt.ylabel(r"irradiance (W.m$^{-2}$)")
    plt.plot(x, intensity_theo, "-", label="Theory")
    plt.plot(x, intensity, "--", label="Beampy")
    plt.legend()
    plt.show()
#    plt.savefig("free_propagation.png", bbox="tight", dpi=250)


def multimodal_splitter():
    """Multimodal splitter 1x2. A single mode beam is split into two single
    mode waveguide by the use of an intermediate multimodal waveguide."""
    width1 = 3
    length1 = 100
    p1 = 4*width1

    width2 = 100
    length2 = 5110
    p2 = 4*width2

    width3 = width1
    length3 = 4090
    p3 = 2*26

    p = [p1, p2, p3]
    nbr_p = [1, 1, 2]

    no = 1.453
    wavelength = 1.55
    delta_no = 0.01
    length_z = length1+length2+length3
    dist_z = 1
    nbr_z_disp = 200
    dist_x = 0.1
    length_x = 1300

    bpm = Bpm(no, wavelength,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape1 = bpm.gauss_guide(width1, 4)
    [peaks, dn] = bpm.create_guides(shape1, delta_no, 1, p1, z=[0, length1])

    shape2 = bpm.squared_guide(width2)
    [peaks2, dn2] = bpm.create_guides(
            shape2, delta_no, 1, p2, z=[length1, length1+length2])

    shape3 = bpm.gauss_guide(width3, 4)
    [peaks3, dn3] = bpm.create_guides(
            shape3, delta_no, 2, p3, z=[length1+length2,
                                        length1+length2+length3])
    dn = np.add(dn, dn2)
    dn = np.add(dn, dn3)
    peaks = np.append(peaks, peaks2, 0)
    peaks = np.append(peaks, peaks3, 0)

    z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)
    xv, zv = np.meshgrid(x, z_disp)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)
    plt.figure("Multimodal beam splitter 1x2")
    plt.title("Multimodal beam splitter 1x2")
    plt.xlabel("x (µm)")
    plt.ylabel("z (mm)")
    plt.pcolormesh(xv, zv, dn[dn_disp], cmap='gray')
    plt.show()
#    plt.savefig("splitter_shape.png", bbox="tight", dpi=250)

    assert bpm.check_modes(width1, delta_no) == 0
    field = bpm.all_modes(width1, delta_no)[0]

    [progress_pow] = bpm.init_field(field, 0, 1)

    [progress_pow] = bpm.main_compute(dn)

    plt.figure("beam profil at the end")
    ax1 = plt.subplot(111)
    plt.title("Beam profil at z=%.0f µm" % length_z)
    plt.xlabel("x (µm)")
    plt.xlim(-p3, p3)
    ax1.set_ylabel(r'$\Delta_n$')
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"I (a.u)")
    ax1.plot(x, dn[-1])
    verts = [(x[0], 0),
             *zip(x, dn[-2, :].real),
             (x[-1], 0)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax1.set_ylim(0, max(dn[0, :].real)*1.1 + 1E-20)
    ax1.add_patch(poly)
    ax2.plot(x, progress_pow[-1]/np.max(progress_pow[0]))
    ax2.set_ylim(0,  1.1)
    plt.show()
#    plt.savefig("splitter_beam_end.png", bbox="tight", dpi=250)

    plt.figure("Propagation in the multimodal beam splitter 1x2")
    plt.title("Beam propagation in the beam splitter 1x2")
    plt.xlabel("x (µm)")
    plt.ylabel("z (mm)")
    plt.pcolormesh(xv, zv, progress_pow)
    plt.show()
#    plt.savefig("splitter_propagation.png", bbox="tight", dpi=250)

    plt.figure("Power in waveguides")
    ax1 = plt.subplot(111)
    ax1.set_title("Power in waveguides")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('z (mm)')
    ax1.set_ylabel('Power (a.u)')

    x_beg = np.array([[None]*nbr_z]*peaks.shape[0])
    x_end = np.array([[None]*nbr_z]*peaks.shape[0])
    P = np.zeros((peaks.shape[0], nbr_z_disp+1))

    num_gd = 0
    for i, n in enumerate(nbr_p):
        for _ in range(n):
            [x_beg[num_gd, :],
             x_end[num_gd, :]] = bpm.guide_position(peaks, num_gd, p[i])
            num_gd += 1
        if n == 0:  # needed if no waveguide
            num_gd += 1

    for i in range(peaks.shape[0]):
        P[i, :] = bpm.power_guide(x_beg[i, :], x_end[i, :])
        ax1.plot(z_disp, P[i, :], label='P'+str(i))
    plt.legend()
    plt.show()
#    plt.savefig("splitter_power.png", bbox="tight", dpi=250)
    print("Input power = %.2f \nOutput powers = %.2f and %.2f" % (P[0, 0],
                                                                  P[2, -1],
                                                                  P[3, -1]))
    print("Losses = %.2f" % (P[0, 0]-P[2, -1]-P[3, -1]))


def benchmark_kerr():
    """Kerr benchmark by looking at the critical power at which a beam become
    a soliton. Several test were done with this function and for now, it seems
    that the critical power find by the simulations is about 85% of the
    theorical value. Meaning a 15% error. Further tests are needed to
    understand the observed differences."""
    width = 0
    no = 1
    wavelength = 4
    delta_no = 0.000
    length_z = 1000
    dist_z = 0.5
    nbr_z_disp = 200
    dist_x = 0.05
    length_x = 500
    nbr_p = 1
    p = 20
    fwhm = 12
    theta_ext = 0
    n2 = 2.44e-20
    alpha = 1.8962  # gaussian beam

    # See Self-focusing on wiki
    P_c = alpha*(wavelength*1e-6)**2/(4*pi*no*n2)
    print("critical power = %.2e W" % P_c)

#    irrad = 5e3*1e13
#    power = irrad * (fwhm*1e-6)**2 * pi / (4*np.log(2))
#    print("power", power)

    # See Gaussian_beam on wiki on beam power
    w = fwhm*1e-6 / sqrt(2*log(2))
    irrad_cri = 2*P_c/(pi*w**2)
    print("Critical irradiance %.2e W/m^2" % irrad_cri)

    bpm = Bpm(no, wavelength,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape = bpm.gauss_guide(width, 4)

    sweep = (1,
             0.2*irrad_cri, 0.5*irrad_cri, 0.7*irrad_cri, 0.75*irrad_cri,
             0.8*irrad_cri, 0.85*irrad_cri,  # 0.85 seem correct
             0.9*irrad_cri, irrad_cri
             )

    for i, irrad in enumerate(sweep):
        power = irrad/2 * (pi*w**2)
        print(i+1, "/", len(sweep), "\tpower= %.2e W" % power, sep="")

        [peaks, dn] = bpm.create_guides(shape, delta_no, nbr_p, p)

        field = bpm.gauss_light(fwhm)

        [progress_pow] = bpm.init_field(field, theta_ext, irrad)

        [progress_pow] = bpm.main_compute(dn, n2=n2, kerr_loop=2,
                                          disp_progress=False)

        x_pos = int(length_x/2/dist_x)
        z_disp = np.linspace(0, length_z/1000, nbr_z_disp+1)

        plt.figure("Benchmark kerr")
        plt.title(r"Central irradition for a theorical $P_c$ = %.2e W" % P_c)
        plt.xlabel("z (µm)")
        plt.ylabel(r"I (a.u)")
#        plt.ylim(-0.05, 2)
        plt.plot(z_disp*1e3,
                 progress_pow[:, x_pos]/progress_pow[0, x_pos],
                 label="Power=%.2e" % power)
        plt.legend()
        plt.show()
#        plt.savefig("kerr_irrad.png", bbox="tight", dpi=250)

#        x_beg = np.array([None]*nbr_z)
#        x_end = np.array([None]*nbr_z)
#        P = np.zeros(nbr_z_disp+1)
#        x_beg, x_end = bpm.guide_position(peaks, 0, p)
#        P = bpm.power_guide(x_beg, x_end)

#        plt.figure("Benchmark kerr2")
#        plt.title("Power for a theorical critical power = %.2e" % P_c)
#        plt.xlabel("z (µm)")
#        plt.ylabel(r"Power (a.u)")
#        plt.ylim(-0.05, 2)
#        plt.plot(z_disp, P, label="Power=%.2e" % power)
#        plt.legend()
#        plt.show()

#        xv, zv = np.meshgrid(x, z_disp)
#        plt.figure("check borders")
#        plt.title("check if beam don't reach the borders")
#        plt.pcolormesh(xv, zv, progress_pow, cmap='gray')
#        plt.show()

        plt.figure("Benchmark beam profil")
        plt.title("Beam profil at z=%.0f µm" % length_z)
        plt.xlabel("x (µm)")
        plt.ylabel(r"I/$I_0$")
        plt.xlim(-30, 30)
        plt.plot(x, progress_pow[-1]/np.max(progress_pow[0]),
                 label="Power=%.2e" % power)
        plt.legend()
        plt.show()
#        plt.savefig("kerr_power.png", bbox="tight", dpi=250)


def stability():
    """Show the possible BPM approximations for implementing a refractive
    index variation"""

    width = 6
    no = 2.14
    wavelength = 1.55
    delta_no = 0.0014
    length_z = 200
    dist_z = 1
    nbr_z_disp = 1
    dist_x = 0.2
    length_x = 1000

    bpm = Bpm(no, wavelength,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    shape = bpm.squared_guide(width)

    nbr_p = 1
    p = 0

    [peaks, dn] = bpm.create_guides(shape, delta_no, nbr_p, p)

    fwhm = 6
    lo = 1.5

    field = bpm.gauss_light(fwhm)
    irrad = 1
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad)

    nbr_step = nbr_z-1

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
    bpm.nl_mat = bpm.ko * bpm.dist_z * dn
    # End overwrite

    field = bpm.field
    for i in range(nbr_step):
        field = ifft(bpm.phase_mat * fft(field))
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
    test1 = field

    field = bpm.field
    for i in range(nbr_step):
        field = ifft(bpm.phase_mat_demi * fft(field))
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
        field = ifft(bpm.phase_mat_demi * fft(field))
    test2 = field

    field = bpm.field
    for i in range(nbr_step):
        field *= np.exp(1j * bpm.nl_mat[nbr_step, :])
        field = ifft(bpm.phase_mat * fft(field))
    test3 = field

    plt.figure("Power")
    plt.title("Power: Impact of the chosen algorithm on the free propagation")
    plt.xlim(-20, 20)
#    plt.ylim(-1, 350)

    plt.plot(x, abs2(test1), label='dz+lens')
    plt.plot(x, abs2(test2), label='dz/2+lens+dz/2')
    plt.plot(x, abs2(test3), label='lens+dz')
    plt.legend()
    plt.show()
    plt.figure("Phase")
    plt.title("Phase: Impact of the chosen algorithm on the free propagation")
    plt.xlim(-30, 30)
    plt.plot(x, np.angle(test1), label='dz+lens')
    plt.plot(x, np.angle(test2), label='dz/2+lens+dz/2')
    plt.plot(x, np.angle(test3), label='lens+dz')
    plt.legend()
    plt.show()

    # Old results showing possibilities to reduce the computation times
#    field = bpm.field
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    test4 = field
#
#    field = bpm.field
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    test5 = field

#    plt.figure("Power 2")
#    plt.title("Power: Same algorithme optimized")
#    plt.xlim(-20, 20)
#    plt.plot(x, abs2(test4), label='dz/2+lens+dz+lens+dz/2')
#    plt.plot(x, abs2(test5), label='(dz/2+lens+dz/2)*2')
#
#    plt.legend()
#    plt.show()
#    plt.figure("Phase 2")
#    plt.title("Phase: Same algorithme optimized")
#    plt.xlim(-30, 30)
#    plt.plot(x, np.angle(test4), label='dz/2+lens+dz+lens+dz/2')
#    plt.plot(x, np.angle(test5), label='(dz/2+lens+dz/2)*2')
#    plt.legend()
#    plt.show()
#
#    field = bpm.field
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat * fft(field))
#    test6 = field
#
#    field = bpm.field
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    field *= np.exp(1j * bpm.nl_mat[0, :])
#    field = ifft(bpm.phase_mat_demi * fft(field))
#    test7 = field
#
#    plt.figure("Power 3")
#    plt.title("Power: Approximation if uses loop over lens+dz")
#    plt.xlim(-20, 20)
#    plt.plot(x, abs2(test6), label='dz/2+lens+dz')
#    plt.plot(x, abs2(test7), label='dz/2+lens+dz/2')
#
#    plt.legend()
#    plt.show()
#    plt.figure("Phase 3")
#    plt.title("Phase: Approximation if uses loop over lens+dz")
#    plt.xlim(-30, 30)
#    plt.plot(x, np.angle(test6), label='dz/2+lens+dz')
#    plt.plot(x, np.angle(test7), label='dz/2+lens+dz/2')
#    plt.legend()
#    plt.show()


def test_kerr():
    """More test than example. Show different approximations for the BPM
    implementation of the Kerr effect."""
    no = 1
    lo = 1.5
    length_z = 50
    dist_z = 0.1
    nbr_z_disp = 1
    dist_x = 0.01
    length_x = 400

    bpm = Bpm(no, lo,
              length_z, dist_z, nbr_z_disp,
              length_x, dist_x)

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()

    dn = np.zeros((nbr_z, nbr_x))

    fwhm = 6

    field = bpm.gauss_light(fwhm)
    irrad = 260e13  # if too high, see big difference between method
    theta_ext = 0

    [progress_pow] = bpm.init_field(field, theta_ext, irrad)

    # Need to overwrite those variables due to changes
    theta = asin(sin(radians(theta_ext)) / no)  # angle in the guide
    nu_max = 1 / (2 * dist_x)  # max frequency due to sampling
    # Spacial frequencies over x (1/µm)
    nu = np.linspace(-nu_max,
                     nu_max*(1 - 2/nbr_x),
                     nbr_x)
    intermed = no * cos(theta) / lo
    fr = -2 * pi * nu**2 / (intermed + np.sqrt(
        intermed**2
        - nu**2
        + 0j))

    bpm.phase_mat = fftshift(np.exp(1j * dist_z * fr))
    bpm.phase_mat_demi = fftshift(np.exp(1j * dist_z / 2 * fr))
    bpm.nl_mat = bpm.ko * bpm.dist_z * dn
    # End overwrite

    kerr_loop = 10
    chi3 = 10 * 1E-20

    nbr_step = nbr_z-1

    print("\n dz/2+lens+dz/2")
    field = bpm.field
    for i in range(nbr_step):
        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        # Influence of the index modulation on the field
        field = field * np.exp(1j * bpm.nl_mat[nbr_step, :])  # No changes if
        # no initial guide (exp=1)

        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        cur_pow = bpm.epnc * abs2(field)

    field_ref = field
    cur_ref = cur_pow

    print("\n dz+kerr")
    field = bpm.field
    for i in range(nbr_step):
        # Linear propagation over dz
        field = ifft(bpm.phase_mat * fft(field))

        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])

        cur_pow = bpm.epnc * abs2(field_x)

        for k in range(1):  # insensitive to correction loop
            prev_pow = cur_pow
            # influence of the beam intensity on the index modulation
            dn_temp = dn[nbr_step, :] + (3*chi3)/(8*no)*(prev_pow/bpm.epnc)
            nl_mat = bpm.ko * bpm.dist_z * dn_temp
            # influence of the index modulation on the field
            field_x = field * np.exp(1j*nl_mat)  # No changes for the pow

            cur_pow = bpm.epnc * abs2(field_x)

        try:
            bpm.variance(prev_pow, cur_pow)
        except ValueError as error:
            print(error)

        field = field_x

    field_1 = field
    cur_1 = cur_pow
    dn_1 = dn_temp

    print("\n dz/2+kerr+dz/2")
    field = bpm.field
    for i in range(nbr_step):
        # Linear propagation over dz/2
        field = ifft(bpm.phase_mat_demi * fft(field))

        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])

        # Linear propagation over dz/2
        field_x = ifft(bpm.phase_mat_demi * fft(field_x))

        cur_pow = bpm.epnc * abs2(field_x)

        for k in range(kerr_loop):
            prev_pow = cur_pow
            # influence of the beam intensity on the index modulation
            dn_temp = dn[nbr_step, :] + (3*chi3)/(8*no)*(prev_pow/bpm.epnc)
            nl_mat = bpm.ko * bpm.dist_z * dn_temp

            # influence of the index modulation on the field
            field_x = field * np.exp(1j * nl_mat)
            # Linear propagation over dz/2
            field_x = ifft(bpm.phase_mat_demi * fft(field_x))

            cur_pow = bpm.epnc * abs2(field_x)

        try:
            bpm.variance(prev_pow, cur_pow)
        except ValueError as error:
            print(error)

        field = field_x

    field_2 = field
    cur_2 = cur_pow
    dn_2 = dn_temp

    print("\n kerr+dz")
    field = bpm.field
    for i in range(nbr_step):
        # Influence of the index modulation on the field
        field_x = field * np.exp(1j * bpm.nl_mat[nbr_step, :])

        # Linear propagation over dz
        field_x = ifft(bpm.phase_mat * fft(field_x))

        cur_pow = bpm.epnc * abs2(field_x)

        for k in range(kerr_loop):
            prev_pow = cur_pow
            # influence of the beam intensity on the index modulation
            dn_temp = dn[nbr_step, :] + (3*chi3)/(8*no)*(prev_pow/bpm.epnc)
            nl_mat = bpm.ko * bpm.dist_z * dn_temp

            # influence of the index modulation on the field
            field_x = field * np.exp(1j * nl_mat)
            # Linear propagation over dz
            field_x = ifft(bpm.phase_mat * fft(field_x))

            cur_pow = bpm.epnc * abs2(field_x)

        try:
            bpm.variance(prev_pow, cur_pow)
        except ValueError as error:
            print(error)

        field = field_x

    field_3 = field
    cur_3 = cur_pow
    dn_3 = dn_temp

    plt.figure(num="Impact order kerr")

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.set_title("phase: comparison")
    ax2.set_title("power: comparison")

    ax1.set_xlim(-15, 15)
    ax2.set_xlim(-15, 15)

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

    dn_ref = dn[nbr_step, :]

    plt.figure(num="Impact on dn order kerr")

    ax1 = plt.subplot(111)

    ax1.set_title("dn: comparison")

    ax1.set_xlim(-15, 15)

    ax1.plot(x, dn_ref, label="no kerr")
    ax1.plot(x, dn_1, label="dz+kerr")
    ax1.plot(x, dn_2, label="dz/2+kerr+dz/2")
    ax1.plot(x, dn_3, label="kerr+dz")

    ax1.legend(loc="upper right")

    plt.show()


def show_grid():
    """Show the computation grid and the displayed grid"""
    length_z = 10
    dist_z = 1
    nbr_z_disp = 6
    length_x = 10
    dist_x = 1
    bpm = Bpm(1, 1, length_z, dist_z, nbr_z_disp, nbr_z_disp, dist_x)
    print("Initial values")
    print("length_z=%f, nbr_z=%f, nbr_z_disp=%f, length_x=%f, nbr_x=%f" % (
            length_z, length_z/dist_z, nbr_z_disp, length_x, length_x/dist_x))

    [length_z, nbr_z, nbr_z_disp, length_x, nbr_x, x] = bpm.create_x_z()
    print("Corrected values")
    print("length_z=%f, nbr_z=%f, nbr_z_disp=%f, length_x=%f, nbr_x=%f" % (
            length_z, nbr_z, nbr_z_disp, length_x, nbr_x))
    dn = np.zeros((nbr_z, nbr_x))

    z = np.linspace(0, length_z, nbr_z)
    z_disp = np.linspace(0, length_z, nbr_z_disp+1)
    dn_disp = np.linspace(0, nbr_z-1, nbr_z_disp+1, dtype=int)
    xv1, zv1 = np.meshgrid(x, z)
    xv2, zv2 = np.meshgrid(x, z_disp)

    plt.figure()
    ax1 = plt.subplot(121)
    ax1.set_title("Computation grid")
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("z (µm)")
    ax1.pcolormesh(xv1, zv1, dn, cmap='gray', edgecolor="w")

    ax1.annotate("", xy=(-dist_x, 1.8*dist_z), xytext=(0., 1.8*dist_z),
                 arrowprops=dict(arrowstyle="<|-|>", connectionstyle="arc3",
                                 color='w'))
    ax1.text(-dist_x/2, 1.3*dist_z, "dist_x",
             {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
              'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    ax1.annotate("", xy=(-1.2*dist_x, 2*dist_z), xytext=(-1.2*dist_x, 3*dist_z),
                 arrowprops=dict(arrowstyle="<|-|>", connectionstyle="arc3",
                                 color='w'))
    ax1.text(-1.5*dist_x, 2.5*dist_z, "dist_z",
             {'color': 'black', 'fontsize': 12, 'ha': 'right', 'va': 'center',
              'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    ax2 = plt.subplot(122)
    ax2.set_title("Displayed grid")
    ax2.set_xlabel("x (µm)")
    ax2.pcolormesh(xv2, zv2, dn[dn_disp], cmap='gray', edgecolor="w")

    ax2.annotate("", xy=(-dist_x, 1.85*bpm.pas), xytext=(0., 1.85*bpm.pas),
                 arrowprops=dict(arrowstyle="<|-|>", connectionstyle="arc3",
                                 color='w'))
    ax2.text(-dist_x/2, 1.6*bpm.pas, "dist_x",
             {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
              'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    ax2.annotate("", xy=(-1.2*dist_x, 2*bpm.pas), xytext=(-1.2*dist_x, 3*bpm.pas),
                 arrowprops=dict(arrowstyle="<|-|>", connectionstyle="arc3",
                                 color='w'))
    ax2.text(-1.5*dist_x, 2.5*bpm.pas, "pas",
             {'color': 'black', 'fontsize': 12, 'ha': 'right', 'va': 'center',
              'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})

    plt.show()
#    plt.savefig("grid_definition.png", bbox="tight", dpi=250)
