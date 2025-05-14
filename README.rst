=============================
obsweatherscale
=============================

.. raw:: html

   <blockquote style="background-color: #f0f0f0; padding: 10px;">
   <strong>⚠️ WARNING</strong><br>
   This project is in BETA and under active development. Interfaces and functionality are subject to change.
   </blockquote>


**obsweatherscale** is a GPyTorch-based Python library for ML probabilistic interpolation and regression using Gaussian Processes (GPs), with a focus on meteorological applications. It provides an extensible framework for building GP models that incorporate neural networks, designed for tasks involving spatial and temporal surface weather analysis fields.

Gaussian Processes are a nonparametric supervised learning method, particularly well-suited for regression tasks that require uncertainty quantification. In our use case, GPs are employed to interpolate observed surface weather variables—such as temperature, precipitation, or wind—recorded at monitoring stations, to arbitrary locations in space. This interpolation is guided by input features, which could include any predictor deemed useful (coordinates, numerical weather prediction (NWP) outputs, topographic information, temporal information, ...).

Possible applications:

- **Downscaling** coarse surface weather analysis fields to fine-scale target grids.
- **Bias correction** of model outputs to better match station observations.
- **Probabilistic interpolation** of observational datasets across space.

Features
--------

- Neural-augmented GP models leveraging trainable mean and kernel functions.
- Plug-and-play data transformations, including normal standardization and quantile fitting.
- Modular architecture for easy experimentation and extension.
- Training routines with support for random masking and batching.
- Inference routines.
- Built on PyTorch for GPU acceleration and automatic differentiation.

Installation
------------
Stable release
~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~
To set up the project for local development, clone the repository and use the provided Poetry setup script:

.. code-block:: console

    $ git clone git@github.com:MeteoSwiss/obsweatherscale.git
    $ cd obsweatherscale
    $ ./scripts/setup_poetry.sh

This will install Poetry (if not already available), set up the virtual environment, and install all dependencies with extras.

Usage
-----

To use obsweatherscale in a project::

    import obsweatherscale


To get started, check out the example scripts in the repository:

* ``example_train_script.py``: Demonstrates how to preprocess data, build a dataset, construct a model with customized mean and kernel functions, and train a GP model using toy data.
* ``example_inference_script.py``: Shows how to perform inference with a trained model to obtain prior and posterior distributions, and how to sample from the distributions.

These examples can be found in the ``obsweatherscale/examples/`` directory and provide a practical introduction to the library's core functionality.

Documentation
-------------
The official documentation is available `here <https://meteoswiss.github.io/obsweatherscale/>`_.

For local development, you can build the documentation using:

.. code-block:: bash

    poetry run sphinx-build doc doc/_build

Then open ``doc/_build/index.html`` in your browser to view the documentation.

Development
-----------
We welcome contributions, suggestions of developments, and bug reports.

Suggestions of developments and bug reports should use the `Issues page of the github repository <https://github.com/MeteoSwiss/obsweatherscale/issues>`_.

Citation
--------

This library is built upon `GPyTorch <https://gpytorch.ai/>`_, which provides the core functionality for Gaussian process modeling and training.  
If you use obsweatherscale in your work, please cite both this library and GPyTorch.

**obsweatherscale**

Lloréns Jover, Icíar and Zanetta, Francesco.  
*obswetherscale: observation-conditioned ML downscaling of surface weather fields.*  
GitHub repository: https://github.com/MeteoSwiss/obsweatherscale,
2025.

.. code-block:: bibtex

    @misc{mch2025yourlib,
      author       = {Lloréns Jover, Icíar and Zanetta, Francesco},
      title        = {obsweatherscale: observation-conditioned ML downscaling of surface weather fields},
      year         = {2025},
      howpublished = {\url{https://github.com/MeteoSwiss/obsweatherscale}},
    }

**GPyTorch**

Gardner, Jacob R., Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson.  
*GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration.*  
In Advances in Neural Information Processing Systems, 2018.

.. code-block:: bibtex

    @inproceedings{gardner2018gpytorch,
      title={GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration},
      author={Gardner, Jacob R and Pleiss, Geoff and Bindel, David and Weinberger, Kilian Q and Wilson, Andrew Gordon},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }


Acknowledgements
----------------

This work benefited from previous research in Gaussian Process modeling for weather data as described in 

License
-------

This project is licensed under the BSD 3-Clause License - see the `LICENSE <https://github.com/MeteoSwiss/obsweatherscale/blob/main/LICENSE>`_ file for details.

Copyright (c) 2024, MeteoSwiss
