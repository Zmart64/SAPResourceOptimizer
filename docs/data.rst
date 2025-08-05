Data Description
================

The project operates on build telemetry exported as a semicolon-separated
CSV file. Each row represents a build or test job. Key raw columns
include:

* ``atom_id`` – UUID of the build job.
* ``time`` – UTC timestamp of the build.
* ``location`` – Cloud location identifier.
* ``memory_fail_count`` – Count of kernel memory allocation failures
  during the build.
* ``branch`` – Source branch that was built.
* ``buildProfile`` – Composite string containing architecture, compiler
  and optimisation level information.
* ``jobs`` – Number of distributed compile jobs.
* ``localJobs`` – Number of local compile jobs on the executor node.
* ``makeType`` – Build type (``dbg``, ``opt`` or ``rel``).
* ``targets`` – CMake targets to build.
* ``component`` – Project component being built.
* ``ts_phase`` – Whether the job belongs to the build or test phase.
* ``ts_status`` – Result status (successful or unstable).
* ``cgroup`` – Executor cgroup UUID.
* ``max_rss`` – Peak RSS memory usage in bytes.
* ``max_cache`` – Peak page cache usage in bytes.
* ``memreq`` – Pre-configured memory request for the build container in
  megabytes.

The :class:`resource_prediction.data_processing.preprocessor.DataPreprocessor`
module derives a rich set of temporal, categorical and rolling window
features from these raw columns.  These engineered features feed the
training and evaluation pipeline documented in :doc:`api`.
