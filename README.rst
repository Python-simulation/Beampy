.. image:: https://readthedocs.org/projects/beampy/badge/?version=latest
   :target: https://beampy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Welcome to Beampy's documentation!
==================================

Beampy is a python module based on the Beam Propagation Method [#bpm]_
used to compute light propagation into a varying refractive index.
The light propagation is done by the bpm module.
An user interface - done using Qt desginer - allows to control the parameters
and display the results.

This project was done by Jonathan Peltier and Marcel Soubkovsky during a master
university course from the PAIP master of the université de Lorraine,
under the directive of Pr. Nicolas Fressengeas.

The bpm codes are mainly based on a compilation of MatLab codes initialy
developed by Régis Grasser during his PhD thesis [#thesis]_,
and later modified at the LMOPS laboratory [#lmops]_.

References
----------

.. [#bpm] K. Okamoto, in Fundamentals of Optical Waveguides,
   2nd ed., edited by K. Okamoto (Academic, Burlington, 2006), pp. 329–397.

.. [#thesis] "Generation et propagation de reseaux periodiques de
   solitons spatiaux dans un milieu de kerr massif" PhD thesis,
   université de Franche-Comté 1998.

.. [#lmops] H. Oukraou et. al., Broadband photonic transport between waveguides
   by adiabatic elimination Phys. Rev. A, 97 023811 (2018).

Links
=====

The online documentation can be found at
`<https://beampy.readthedocs.io/>`_.

The source code of the whole project can be found at
`<https://github.com/Python-simulation/Beampy/>`_.

The PyPI repository can be found at `<https://pypi.org/project/beampy/>`_.


Installation
============

This package can be download in a python environment using pip install::

    pip install beampy

Or by downloading the github folder and setting beampy as a PYTHONPATH.
If so, make sure to download Qt5, matplotlib and numpy.


Starting the software
=====================

To start the Beampy interface, import beampy and start the open_app function::

    import beampy
    beampy.open_app()

Or open direclty the user_interface.py file to launch the interface.
Or even open the bpm.py to have a non-based interface version.
