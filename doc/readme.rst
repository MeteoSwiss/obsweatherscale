.. This file is auto-generated. Do not edit.

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

Features:

- Neural-augmented GP models leveraging trainable mean and kernel functions.
- Plug-and-play data transformations, including normal standardization and quantile fitting.
- Modular architecture for easy experimentation and extension.
- Training routines with support for random masking and batching.
- Inference routines.
- Built on PyTorch for GPU acceleration and automatic differentiation.
