# KEY INSIGHTS FROM EDA
1. Memory Usage Range: 28.90 MB to 236,322 MB (very wide range)
2. Success Rate: 68.85% of builds are successful
3. Memory Failure Rate: 11% of builds have memory failures
4. Peak Build Hours: 9 AM, 4 PM, and 11 PM
5. Weekday Pattern: More builds on Tuesday-Thursday
6. Job Distribution: Mostly 500 jobs (69%), some 1000 (28%), few 2000 (3%)
7. Build Profiles: gcc-optimized is most common (38%)
8. Components: component_1 dominates (80% of builds)


# ADVANCED INSIGHTS
1. STRONGEST PREDICTORS:
   - Jobs (r=0.24): More jobs → higher memory usage
   - Max Cache (r=0.22): Larger cache → higher RSS
   - Jobs per Local (r=0.18): Higher ratio → more memory

2. BUILD PROFILE INSIGHTS:
   - clang_asan-optimized: Highest memory (109GB avg) - Address Sanitizer overhead
   - gcc-optimized: Most common (38%) but lower memory (58GB avg)
   - gcc-release: Higher memory (85GB) but better success rate (83%)

3. TEMPORAL PATTERNS:
   - Peak hours: 9 AM, 4 PM, 11 PM
   - Weekdays dominate: Tue-Thu have most builds
   - Memory usage varies by time of day

4. FAILURE PATTERNS:
   - 11% of builds have memory failures
   - Memory failures correlate weakly with memory usage (r=0.07)
   - Some branches have much higher failure rates

5. RESOURCE UTILIZATION:
   - Memory efficiency varies widely (0.65 median, 8.5 std)
   - Jobs/LocalJobs ratio: 20.8 average (high parallelization)
   - Component_1 dominates (80% of builds)

# REALLY ADVANCED INSIGHTS
1. DATASET CHARACTERISTICS:
   • Total builds: 22,990
   • Time span: 13 days
   • Unique branches: 457
   • Unique build profiles: 56
   • Memory range: 28.9 - 236322.0 MB

2. TARGET VARIABLE (max_rss) DISTRIBUTION:
   • Mean: 72566.2 MB
   • Median: 71180.7 MB
   • Std: 36056.5 MB
   • Skewness: 0.61 (right-skewed)
   • CV: 0.50 (high variability)

3. FEATURE ENGINEERING OPPORTUNITIES (from paper):
   • Historical features: Previous builds' memory usage
   • Temporal features: Hour, day of week, time since last build
   • Resource ratios: jobs/localJobs, memory_efficiency
   • Categorical encoding: Build profiles, branches, components
   • Interaction features: Profile × Jobs, Branch × Component

4. PREDICTIVE FEATURES IDENTIFIED:
   Top correlations with memory usage:
     - jobs: 0.240
     - max_cache_mb: 0.221
     - jobs_per_local: 0.180
     - memreq_mb: 0.122
     - localJobs: 0.078
     - memory_fail_count: 0.074

5. DATA QUALITY ASSESSMENT:
   • Missing values: 0 (none)
   • Successful builds: 68.9%
   • Memory failures: 11.0%
   • Outliers (>3σ): 0.5%

6. CATEGORICAL FEATURE ANALYSIS:
   • buildProfile: 56 categories, top category: 38.5%
   • branch: 457 categories, top category: 18.8%
   • makeType: 3 categories, top category: 61.4%
   • component: 2 categories, top category: 80.3%
   • location: 4 categories, top category: 46.6%
   • targets: 47 categories, top category: 77.2%

7. TEMPORAL PATTERNS FOR FEATURE ENGINEERING:
   • Peak build hours: 9 AM, 4 PM, 11 PM (workday patterns)
   • Weekday bias: More builds Tue-Thu (development cycles)
   • Daily memory variation: Potential for time-based features

8. RESOURCE UTILIZATION INSIGHTS:
   • Job parallelization: 20.8x average
   • Memory efficiency spread: 8.5 (high variance)
   • Cache utilization: Strong predictor of RSS usage

9. RECOMMENDATIONS FOR ML PIPELINE:
   • Target transformation: Log or Box-Cox (right-skewed distribution)
   • Feature scaling: StandardScaler for numeric features
   • Categorical encoding: Target encoding for high-cardinality features
   • Cross-validation: Time-based splits (temporal dependency)
   • Model selection: Tree-based models (handle non-linearity)
   • Evaluation metrics: RMSE, MAE, MAPE (regression problem)

10. POTENTIAL CHALLENGES:
   • High cardinality: 457 branches, 56 build profiles
   • Imbalanced success rates: Only 68.85% successful builds
   • Wide memory range: 8000x difference between min/max
   • Temporal dependencies: Sequential build relationships


# Comments
1. There are build with 500, 1000 and 2000 jobs
2. What specifically are "local jobs"
3. Too much memory is being requested:


This needs to investiagted
--- MEMORY REQUEST vs ACTUAL USAGE ---
Memory request correlation with actual usage: 0.122
Average memory overuse ratio: 422024.250
Median memory overuse ratio: 399182.900
Memory overuse distribution:
  < 0.5x requested: 0.0%
  0.5x - 1.0x:      0.0%
  1.0x - 2.0x:      0.0%
  > 2.0x requested: 100.0%
