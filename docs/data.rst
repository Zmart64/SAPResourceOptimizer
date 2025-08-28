Data Description
================

The project operates on build telemetry exported as a semicolon-separated
CSV file. Each row represents a build or test job from a distributed CI/CD
system. The goal is to predict optimal memory allocation for future jobs
to minimize both build failures and resource waste.

Business Context
----------------

In large-scale software development, distributed build systems process
thousands of compilation jobs daily. Each job requires memory allocation
upfront, creating a critical optimization problem:

- **Under-allocation**: Causes out-of-memory failures, wasting developer time and CI resources
- **Over-allocation**: Leads to resource waste and increased infrastructure costs
- **Temporal Dependencies**: Build patterns change over time with code evolution

Raw Data Schema
---------------

Key raw columns include:

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
* ``max_rss`` – Peak RSS memory usage in bytes (target variable).
* ``max_cache`` – Peak page cache usage in bytes.
* ``memreq`` – Pre-configured memory request for the build container in
  megabytes.

Feature Engineering Pipeline
----------------------------

The :class:`resource_prediction.data_processing.preprocessor.DataPreprocessor`
derives a rich set of temporal, categorical and rolling window
features from these raw columns. These engineered features feed the
training and evaluation pipeline documented in :doc:`api`.

**Base Features** (always included):

- **Temporal**: ``ts_year``, ``ts_month``, ``ts_dow``, ``ts_hour``, ``ts_weekofyear``
- **Categorical**: ``location``, ``component``, ``makeType``, ``bp_arch``, ``bp_compiler``, ``bp_opt``
- **Derived**: ``branch_prefix``, ``target_cnt``, ``target_has_dist``, ``branch_id_str``
- **Historical**: ``lag_1_grouped``, ``lag_max_rss_global_w5``, ``rolling_p95_rss_g1_w5``
- **Parallelism**: ``jobs``, ``localJobs``

**Quantitative Features** (optional):

- ``build_load`` – Normalized measure of system load during build
- ``target_intensity`` – Complexity score based on target composition
- ``debug_multiplier`` – Memory scaling factor for debug builds
- ``heavy_target_flag`` – Binary indicator for memory-intensive targets
- ``high_parallelism`` – Flag for highly parallel build configurations

Target Variable Processing
--------------------------

- Raw ``max_rss`` (bytes) → ``max_rss_gb`` (gigabytes)
- Time-series split: 90% training, 10% holdout testing
- For classification: Dynamic quantile binning to handle varying memory distributions

Data Quality Considerations
---------------------------

- Handles missing values and outliers in preprocessing
- Respects temporal ordering for cross-validation splits
- Categorical encoding with consistent feature spaces across train/test
- Rolling window features capture historical build patterns
