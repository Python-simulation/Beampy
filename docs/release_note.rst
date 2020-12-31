Release note
============

Todo (not sure)
---------------

Change the whole code structure to have a more readable and convinient one
with a windows, waveguide, beam, propagation distinct class.

Add an autosave with possibility to go back to previous parameters choices.

Fixe the plotting issue when using tight layout.


Version 1.11
------------

bpm.py and user_interface.py
````````````````````````````

Waveguides can have a finite z dimension.

Remove the losses menu to incorporate the losses in the refractive index.

Correct an error in the Kerr calculation. Previous results were wrong by a
factor 2eta/no = 753/no.

Add the nonlinear refractive index n2 as a alternative to chi3.

Change most of the operations into numpy ones to speed up the computation.

Add a option to select which waveguide width and dn are used to find the
corresponding optical mode.

Copy the main_compute function into the user_interface to be able to display
the computation progression onto the interface.

The power in each waveguides can be computed and displayed without recomputing
the whole propagation.

Variables chi3, n2 and irrad are now defined by a significand and a exponent,
allowing to choose a larger range in the interface version.

Correct an error in the power calculation for curved waveguides. Previous
results were wrong for high curvature factor.

Change the linestyle in the power display to have more control over it.

examples.py
```````````

Update examples.

Add a multimodal beam splitter example.

Add an attempt of benchmarking the Kerr effect.

Version 1.1
-----------

Implement a waveguide creation menu based on the existing beam menu.
It is now possible to create as many waveguides as wanted, with different
parameters for each one.
