=======
History
=======

Unreleased
----------

* Breaking: ``Trainer`` no longer takes a ``likelihood`` argument (it is
  taken from ``model.likelihood``), and ``Trainer.fit`` returns the trainer
  instead of a ``(model, progress)`` tuple. Per-iteration metrics are in
  ``Trainer.history``, best weights in ``Trainer.best_model``, and
  ``self.model`` holds the best weights after ``fit``.
* Breaking: ``nan_policy`` is now an attribute of ``GPModel`` (default
  ``"fill"``) instead of a ``Trainer.fit`` argument.
* Breaking: ``make_mll_loss`` and ``make_crps_loss`` take the model instead of
  an MLL object / likelihood. Added a ``make_loss`` factory.
* ``Trainer.fit`` saves a checkpoint per iteration (``model_{i}.pt``) when
  ``output_dir`` is set, and no longer closes the loggers.
* ``MLflowLogger`` supports resuming a child run via ``run_id``; fixed run
  management bugs.
* Refactored data transformations; added docstrings across the package.

0.1.0 (2025-10-04)
------------------

* First release.
