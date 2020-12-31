"""Beampy is a module based on Beam Propagation Method, a numerical method
used to compute light propagation into a refractive index.
The light propagation is done by the :mod:`beampy.bpm` module, shown and
piloted by the :mod:`beampy.user_interface` module.
This user interface call the :mod:`interface` module created with Qt desginer.

Modules include:

    :mod:`beampy.bpm`
        defines the :class:`beampy.bpm.Bpm` class.

    :mod:`beampy.user_interface`
        defines the :class:`beampy.user_interface.UserInterface` class.

    :mod:`beampy.interface`
        defines the :class:`beampy.interface.setupUi` class.
        And also define the unused :class:`retranslateUi`

This project was initiate by Jonathan Peltier and Marcel Soubkovsky during
a master university course from the PAIP master of the université de Lorraine,
under the directive of Pr. Nicolas Fressengeas.

The bpm codes are mainly based on a compilation of MatLab codes initialy
developed by Régis Grasser during his PhD thesis,
and later modified at the LMOPS laboratory.
"""
from beampy.user_interface import (open_app, open_doc)  # was accessible with
# user_interface but it's best to have those functions right in the beampy
# module
from beampy import examples

__version__ = "1.11"


def help():
    print("Use the open_app function to launch the beampy app.")
    print("Possible way to do so: beampy.open_app(),")
    print("Or open the user_interface.py")
    print("\n")
    print("For more help, open the documentation with:")
    print("beampy.open_doc() or")
    print("\n")
    print("Examples can be found using beampy.examples")
    print("and the example name, for example: beampy.examples.gaussian_beam()")
    print("\n")
    print("All the documentation can be found on the website:")
    print("https://beampy.readthedocs.io")


if __name__ == "__main__":
    open_app()
