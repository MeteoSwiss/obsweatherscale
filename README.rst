=============================
obsweatherscale
=============================

Python library for ML-based downscaling of surface weather analysis fields, with conditioning on in-situ observations.

Setup virtual environment

.. code-block:: console

    $ cd obsweatherscale
    $ poetry install


Run tests

.. code-block:: console

    $ poetry run pytest

Generate documentation

.. code-block:: console

    $ poetry run sphinx-build doc doc/_build

Then open the index.html file generated in *obsweatherscale/build/_build/*

Build wheels

.. code-block:: console

    $ poetry build
