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

This software was mainly developed by Jonathan Peltier and, the interface was
firstly developed by Marcel Soubkovsky.

This project was part of a university course, under the direction of Pr.
Nicolas Fressengeas.

At first, the objective was to translate BPM codes - used in the LMOPS
laboratory - done in MatLab into Python codes.
Then, an interface were added to make the programs more accessible to people
whom don't know python or don't have a sufficient coding baggage.
Finaly, more options were added to completed the programs.
"""
from beampy.user_interface import (open_app, open_doc)  # was accessible with
# user_interface but it's best to have those functions right in the beampy
# module
from beampy import examples

__version__ = "0.1.0"


def help():
    print("Use the open_app function to launch the beampy app.")
    print("Possible way to do so: beampy.open_app(),")
    print("beampy.user_interface.open_app().")
    print("Or open the user_interface.py")
    print("or __init__.py file from the beampy module")
    print("\n")
    print("For more help, open the documentation with:")
    print("beampy.open_doc() or")
    print("with beampy.user_interface.open_doc()")
    print("\n")
    print("Examples can be found using beampy.examples")
    print("and the example name, for example: .example_beams()")
    print("\n")
    print("All the documentation can be found on the site:")
    print("https://beampy.readthedocs.io")


if __name__ == "__main__":
    open_app()
