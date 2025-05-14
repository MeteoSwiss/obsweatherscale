.. highlight:: shell

============
Installation
============


Stable release
--------------

To install obsweatherscale, run this command in your terminal:

.. code-block:: console

    $ pip install obsweatherscale

This is the preferred method to install obsweatherscale, as it
will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


For development
--------------

To set up the project for local development, clone the repository and use the 
provided Poetry setup script:

.. code-block:: console

    $ git clone git@github.com:MeteoSwiss/obsweatherscale.git
    $ cd obsweatherscale
    $ ./scripts/setup_poetry.sh

This will install Poetry (if not already available), set up the virtual 
environment, and install all dependencies with extras.