.. title:: Installation


============
Installation
============

FIAT is normally installed as part of an installation of FEniCS.
If you are using FIAT as part of the FEniCS software suite, it
is recommended that you follow the
`installation instructions for FEniCS
<https://fenics.readthedocs.io/en/latest/>`__.

To install FIAT itself, read on below for a list of requirements
and installation instructions.


Requirements and dependencies
=============================

FIAT requires Python version 2.7 or later and depends on the
following Python packages:

* NumPy
* SymPy
* six

These packages will be automatically installed as part of the
installation of FIAT, if not already present on your system.


Installation instructions
=========================

To install FIAT, download the source code from the
`FIAT Bitbucket repository
<https://bitbucket.org/fenics-project/fiat>`__,
and run the following command:

.. code-block:: console

    pip install .

To install to a specific location, add the ``--prefix`` flag
to the installation command:

.. code-block:: console

    pip install --prefix=<some directory> .
