"""EDA"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/build-data.csv", delimiter=";")

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")

# Convert memory values to MB for better readability
df["max_rss_mb"] = df["max_rss"] / (1024**2)
df["max_cache_mb"] = df["max_cache"] / (1024**2)
df["memreq_mb"] = df["memreq"] / (1024**2)

print("\n=== TARGET VARIABLE ANALYSIS (max_rss in MB) ===")
print(f"Min: {df['max_rss_mb'].min():.2f} MB")
print(f"Max: {df['max_rss_mb'].max():.2f} MB")
print(f"Mean: {df['max_rss_mb'].mean():.2f} MB")
print(f"Median: {df['max_rss_mb'].median():.2f} MB")
print(f"Std: {df['max_rss_mb'].std():.2f} MB")


#####################################################
# Basic Distribution Analysis of max_rss
# Histogram with mean and median lines

fig, axes = plt.subplots(1, 2, figsize=(15, 12))
axes[0].hist(df["max_rss_mb"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
axes[0].axvline(
    df["max_rss_mb"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {df['max_rss_mb'].mean():.1f} MB",
)
axes[0].axvline(
    df["max_rss_mb"].median(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {df['max_rss_mb'].median():.1f} MB",
)
axes[0].set_xlabel("max_rss (MB)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("max_rss distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Box plot
box_plot = axes[1].boxplot(df["max_rss_mb"], patch_artist=True)
box_plot["boxes"][0].set_facecolor("lightgreen")
axes[1].set_ylabel("max_rss in MB")
axes[1].set_title("max_rss Box Plot")
axes[1].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/00_max_rss_analysis.png",
    dpi=300,
)
plt.close()


#####################################################
# CDF of max_rss_mb
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

sorted_memory = np.sort(df["max_rss_mb"])
y_values = np.arange(1, len(sorted_memory) + 1) / len(sorted_memory)

ax.plot(sorted_memory, y_values, linewidth=2, color="steelblue")
ax.set_xlabel("Memory Usage (MB)")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Cumulative Distribution Function of Memory Usage")
ax.grid(True, alpha=0.3)
ax.set_xscale("log")

# Add percentile markers
percentiles = [50, 90, 95, 99]
for p in percentiles:
    value = np.percentile(df["max_rss_mb"], p)
    ax.axvline(value, color="red", linestyle="--", alpha=0.7)
    ax.text(
        value,
        p / 100,
        f"{p}th: {value:.0f}MB",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/01_max_rss_cdf.png",
    dpi=300,
)
plt.close()


print("\n=== CATEGORICAL VARIABLES ANALYSIS ===")
categorical_cols = [
    "location",
    "branch",
    "buildProfile",
    "makeType",
    "targets",
    "component",
    "ts_phase",
    "ts_status",
]

for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"\n{col}: {unique_count} unique values")
    if unique_count <= 10:
        print(df[col].value_counts().head(10))
    else:
        print(df[col].value_counts().head(5))

print("\n=== MEMORY FAILURE ANALYSIS ===")
print("Memory failure distribution:")
print(df["memory_fail_count"].value_counts().head(10))
print(f"Builds with memory failures: {(df['memory_fail_count'] > 0).sum()}")
print(f"Memory failure rate: {(df['memory_fail_count'] > 0).mean() * 100:.2f}%")

print("\n=== TIME-BASED ANALYSIS ===")
df["time"] = pd.to_datetime(df["time"], format="mixed")
df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek
df["date"] = df["time"].dt.date

print(f"Time range: {df['time'].min()} to {df['time'].max()}")
print(f"Total days: {(df['time'].max() - df['time'].min()).days}")

print("\n=== BUILD ANALYSIS ===")

print("Build status distribution:")
print(df["ts_status"].value_counts())
print(f"\nSuccess rate: {(df['ts_status'] == 'ok').mean() * 100:.2f}%")

print("\nBuilds per hour:")
hourly_builds = df["hour"].value_counts().sort_index()
print(hourly_builds)

print("\nBuilds per day of week (0=Monday, 6=Sunday):")
daily_builds = df["day_of_week"].value_counts().sort_index()
print(daily_builds)

# Top 20 build profiles
profile_stats = (
    df.groupby("buildProfile")
    .agg(
        {
            "max_rss_mb": ["count", "mean", "std", "min", "max"],
            "ts_status": lambda x: (x == "ok").mean(),
        }
    )
    .round(2)
)

profile_stats.columns = [
    "count",
    "mean_memory",
    "std_memory",
    "min_memory",
    "max_memory",
    "success_rate",
]
profile_stats = profile_stats.sort_values("count", ascending=False)
print("\nTop 20 build profiles by frequency:")
print(profile_stats.head(20))


# Split buildProfile into components
df["architecture"] = df["buildProfile"].str.extract(r"(linux\w+)")
df["compiler"] = df["buildProfile"].str.extract(r"-(gcc\d*)")
df["optimization"] = df["buildProfile"].str.extract(r"-(optimized|debug|release)")

df["architecture"] = df["architecture"].fillna("unknown")
df["compiler"] = df["compiler"].fillna("unknown")
df["optimization"] = df["optimization"].fillna("unknown")

print("\nArchitecture impact:")
arch_stats = (
    df.groupby("architecture").agg({"max_rss_mb": ["count", "mean", "std"]}).round(3)
)
arch_stats.columns = ["count", "mean_memory", "std_memory"]
print(arch_stats.sort_values("mean_memory", ascending=False))

print("\nCompiler impact:")
compiler_stats = (
    df.groupby("compiler").agg({"max_rss_mb": ["count", "mean", "std"]}).round(3)
)
compiler_stats.columns = ["count", "mean_memory", "std_memory"]
print(compiler_stats.sort_values("mean_memory", ascending=False))

print("\nOptimization level impact:")
opt_stats = (
    df.groupby("optimization").agg({"max_rss_mb": ["count", "mean", "std"]}).round(3)
)
opt_stats.columns = ["count", "mean_memory", "std_memory"]
print(opt_stats.sort_values("mean_memory", ascending=False))

print("\nMake Type impact")
maketype_stats = (
    df.groupby("makeType")
    .agg(
        {
            "max_rss_mb": ["count", "mean", "std", "median"],
            "ts_status": lambda x: (x == "ok").mean(),
        }
    )
    .round(3)
)
maketype_stats.columns = [
    "count",
    "mean_memory",
    "std_memory",
    "median_memory",
    "success_rate",
]
print(maketype_stats.sort_values("mean_memory", ascending=False))

#####################################################
# Plot build related information

# Memory usage by build profile
# There are 56 different build profiles, so we will focus on the top 10 most frequent ones
plt.figure(figsize=(12, 8))
top_profiles = df["buildProfile"].value_counts().head(10).index
df_top_profiles = df[df["buildProfile"].isin(top_profiles)]
df_top_profiles.boxplot(column="max_rss_mb", by="buildProfile")
plt.xticks(rotation=45, ha="right")
plt.ylabel("max_rss in MB")
plt.title("Memory Usage by Top 10 Build Profile")
plt.suptitle("")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/02_max_rss_by_build_profile.png"
)
plt.close()

# Memory usage by build status
plt.figure(figsize=(8, 6))
df.boxplot(column="max_rss_mb", by="ts_status")
plt.xticks(rotation=45)
plt.ylabel("max_rss in MB")
plt.title("Memory Usage by Build Status")
plt.suptitle("")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/03_max_rss_by_build_status.png"
)
plt.close()

# Builds over time
plt.figure(figsize=(12, 6))
daily_counts = df.groupby("date").size()
plt.plot(daily_counts.index, daily_counts.values, marker="o", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Number of Builds")
plt.title("Builds Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/04_builds_over_time.png")
plt.close()

# Hourly build pattern
plt.figure(figsize=(10, 6))
hourly_builds.plot(kind="bar", color="lightgreen", alpha=0.7)
plt.xlabel("Hour of Day")
plt.ylabel("Number of Builds")
plt.title("Build Activity by Hour")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/05_hourly_builds.png")
plt.close()

# Build phase comparison
plt.figure(figsize=(10, 6))
phase_data = [
    df[df["ts_phase"] == phase]["max_rss_mb"].values
    for phase in df["ts_phase"].unique()
]
bp = plt.boxplot(phase_data, tick_labels=df["ts_phase"].unique(), patch_artist=True)
colors = ["lightblue", "lightcoral", "lightgreen"][: len(bp["boxes"])]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
plt.ylabel("max_rss (MB)")
plt.title("max_rss by Build Phase")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/06_memory_by_phase.png")
plt.close()


# Make type analysis
plt.figure(figsize=(8, 6))
df.boxplot(column="max_rss_mb", by="makeType")
plt.title("max_rss by Make Type")
plt.ylabel("max_rss in MB")
plt.suptitle("")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/07_max_rss_by_make_type.png"
)
plt.close()

# max_rss by architecture, compiler, and optimization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Architecture
arch_data = [
    df[df["architecture"] == arch]["max_rss_mb"].values
    for arch in df["architecture"].unique()
]
axes[0].boxplot(arch_data, tick_labels=df["architecture"].unique())
axes[0].set_title("max_rss by Architecture")
axes[0].set_ylabel("max_rss (MB)")
axes[0].tick_params(axis="x", rotation=45)

# Compiler
comp_data = [
    df[df["compiler"] == comp]["max_rss_mb"].values for comp in df["compiler"].unique()
]
axes[1].boxplot(comp_data, tick_labels=df["compiler"].unique())
axes[1].set_title("max_rss by Compiler")
axes[1].set_ylabel("max_rss (MB)")
axes[1].tick_params(axis="x", rotation=45)

# Optimization
opt_data = [
    df[df["optimization"] == opt]["max_rss_mb"].values
    for opt in df["optimization"].unique()
]
axes[2].boxplot(opt_data, tick_labels=df["optimization"].unique())
axes[2].set_title("max_rss by Optimization Level")
axes[2].set_ylabel("max_rss (MB)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/08_max_rss_by_build_components_detailed.png"
)
plt.close()

print("\n=== BRANCH ANALYSIS ===")
branch_stats = (
    df.groupby("branch")
    .agg(
        {
            "max_rss_mb": ["count", "mean"],
            "ts_status": lambda x: (x == "ok").mean(),
        }
    )
    .round(2)
)
branch_stats.columns = ["count", "mean_memory", "success_rate"]
branch_stats = branch_stats.sort_values("count", ascending=False)
print("Top 10 branches by frequency:")
print(branch_stats.head(10))

# Top 10 branches memory usage boxplot
plt.figure(figsize=(14, 8))
top_10_branches = df["branch"].value_counts().head(10).index
branch_memory_data = [
    df[df["branch"] == branch]["max_rss_mb"].values for branch in top_10_branches
]
bp = plt.boxplot(branch_memory_data, tick_labels=top_10_branches, patch_artist=True)
colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, len(bp["boxes"])))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
plt.ylabel("max_rss (MB)")
plt.title("max_rss by Branch (Top 10 by Frequency)")
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/09_max_rss_by_top_10_branches.png"
)
plt.close()

# isnt really useful
# # max_rss vs success rate by branch
# plt.figure(figsize=(10, 6))
# branch_counts = df.groupby("branch").size()
# branch_analysis = (
#     df.groupby("branch")
#     .agg({"max_rss_mb": "mean", "ts_status": lambda x: (x == "ok").mean()})
#     .reset_index()
# )
# branch_analysis = branch_analysis[
#     branch_analysis["branch"].isin(branch_counts[branch_counts >= 50].index)
# ]

# # Create colors for each branch
# colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, len(branch_analysis)))
# for i, (_, row) in enumerate(branch_analysis.iterrows()):
#     plt.scatter(
#         row["max_rss_mb"],
#         row["ts_status"],
#         alpha=0.7,
#         s=50,
#         c=[colors[i]],
#         label=row["branch"],
#     )

# plt.xlabel("Average max_rss (MB)")
# plt.ylabel("Success Rate")
# plt.title("max_rss vs Success Rate by Branch")
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.savefig(
#     "/home/krebs/Distributed_Systems_Project/eda/plots/09_memory_vs_success_rate.png"
# )
# plt.close()

print("\n=== LOCATION ANALYSIS ===")
location_stats = (
    df.groupby("location")
    .agg(
        {
            "max_rss_mb": ["count", "mean", "std"],
            "ts_status": lambda x: (x == "ok").mean(),
        }
    )
    .round(2)
)
location_stats.columns = [
    "count",
    "mean_memory",
    "std_memory",
    "success_rate",
]
print("Location statistics:")
print(location_stats)


# max_rss by location
plt.figure(figsize=(12, 8))
location_data = [
    df[df["location"] == loc]["max_rss_mb"].values for loc in df["location"].unique()
]
bp = plt.boxplot(location_data, tick_labels=df["location"].unique(), patch_artist=True)
colors = plt.colormaps.get_cmap("Set3")(np.linspace(0, 1, len(bp["boxes"])))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
plt.ylabel("max_rss (MB)")
plt.title("max_rss Distribution by Location")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/10_max_rss_by_location.png"
)
plt.close()

# max_rss and success rate by location
plt.figure(figsize=(12, 8))
location_analysis = (
    df.groupby("location")
    .agg({"max_rss_mb": "mean", "ts_status": lambda x: (x == "ok").mean()})
    .reset_index()
)
scatter = plt.scatter(
    location_analysis["max_rss_mb"], location_analysis["ts_status"], s=100, alpha=0.7
)
for i, location in enumerate(location_analysis["location"]):
    plt.annotate(
        location,
        (
            location_analysis["max_rss_mb"].iloc[i],
            location_analysis["ts_status"].iloc[i],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
plt.xlabel("Average max_rss (MB)")
plt.ylabel("Success Rate")
plt.title("max_rss vs Success Rate by Location")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/11_max_rss_and_success_by_location.png"
)
plt.close()

print("\n--- TARGETS ANALYSIS ---")
targets_counts = df["targets"].value_counts()
top_targets = targets_counts.head(10).index
targets_stats = (
    df[df["targets"].isin(top_targets)]
    .groupby("targets")
    .agg({"max_rss_mb": ["count", "mean", "std"]})
    .round(3)
)
targets_stats.columns = ["count", "mean_memory", "std_memory"]
print("Top 10 targets by frequency:")
print(targets_stats.sort_values("mean_memory", ascending=False))

# Impact of the target on max_rss
plt.figure(figsize=(14, 8))
top_10_targets = df["targets"].value_counts().head(10).index
target_memory_data = [
    df[df["targets"] == target]["max_rss_mb"].values for target in top_10_targets
]
plt.boxplot(
    target_memory_data,
    tick_labels=[t[:20] + "..." if len(t) > 20 else t for t in top_10_targets],
)
plt.ylabel("max_rss (MB)")
plt.title("max_rss by Build Target (Top 10)")
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/12_max_rss_by_targets.png"
)
plt.close()

print("\n--- COMPONENT ANALYSIS ---")
component_stats = (
    df.groupby("component")
    .agg(
        {
            "max_rss_mb": ["count", "mean", "std", "median"],
            "ts_status": lambda x: (x == "ok").mean(),
        }
    )
    .round(3)
)
component_stats.columns = [
    "count",
    "mean_memory",
    "std_memory",
    "median_memory",
    "success_rate",
]
print(component_stats.sort_values("mean_memory", ascending=False))


plt.figure(figsize=(10, 6))
component_data = [
    df[df["component"] == comp]["max_rss_mb"].values
    for comp in df["component"].unique()
]
plt.boxplot(component_data, tick_labels=df["component"].unique())
plt.ylabel("max_rss (MB)")
plt.title("max_rss by Component")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/14_max_rssy_by_component.png"
)
plt.close()


print("\n=== MEMORY USAGE ANALYSIS ===")

print("\nMemreq debugging:")
print(f"memreq min: {df['memreq'].min()}")
print(f"memreq max: {df['memreq'].max()}")
print(f"memreq mean: {df['memreq'].mean()}")
print(f"memreq zero count: {(df['memreq'] == 0).sum()}")
print(f"memreq non-zero count: {(df['memreq'] > 0).sum()}")
print(f"memreq_mb min: {df['memreq_mb'].min()}")
print(f"memreq_mb max: {df['memreq_mb'].max()}")
print(f"memreq_mb mean: {df['memreq_mb'].mean()}")


# Time series of memory usage
plt.figure(figsize=(12, 6))
daily_memory = df.groupby("date")["max_rss_mb"].mean()
daily_memreq = df.groupby("date")["memreq_mb"].mean()
print(
    f"Daily memory usage range: {daily_memory.min():.2f} MB to {daily_memory.max():.2f} MB"
)
print(f"Daily memreq range: {daily_memreq.min():.2f} MB to {daily_memreq.max():.2f} MB")

# Plot all days in sequence
plt.plot(
    range(len(daily_memory)),
    daily_memory.values,
    marker="o",
    linewidth=2,
    label="max_rss",
    color="blue",
)
plt.plot(
    range(len(daily_memreq)),
    daily_memreq.values,
    marker="s",
    linewidth=2,
    label="memreq",
    color="red",
)
plt.xlabel("Days")
plt.ylabel("Average Memory (MB)")
plt.title("Daily Average Memory Usage and Requests Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/15_daily_max_rss.png")
plt.close()

# # Hourly memory usage pattern
# plt.figure(figsize=(10, 6))
# hourly_memory = df.groupby("hour")["max_rss_mb"].mean()
# hourly_memreq = df.groupby("hour")["memreq_mb"].mean()
# plt.plot(
#     hourly_memory.index,
#     hourly_memory.values,
#     marker="o",
#     linewidth=2,
#     color="orange",
#     label="max_rss",
# )
# plt.plot(
#     hourly_memreq.index,
#     hourly_memreq.values,
#     marker="s",
#     linewidth=2,
#     color="green",
#     label="memreq",
# )
# plt.xlabel("Hour of Day")
# plt.ylabel("Average Memory (MB)")
# plt.title("Hourly Memory Usage and Requests")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/16_hourly_max_rss.png")
# plt.close()

# Memory failures vs max_rss
plt.figure(figsize=(10, 6))
failed_builds = df[df["memory_fail_count"] > 0]
successful_builds = df[df["memory_fail_count"] == 0]
plt.hist(
    [successful_builds["max_rss_mb"], failed_builds["max_rss_mb"]],
    bins=30,
    alpha=0.7,
    label=["No Failures", "With Failures"],
    color=["green", "red"],
)
plt.xlabel("max_rss (MB)")
plt.ylabel("Frequency")
plt.title("max_rss: Failed vs Successful Builds")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/17_memory_failures.png")
plt.close()


# Memory cache vs max_rss
plt.figure(figsize=(10, 6))
plt.scatter(df["max_cache_mb"], df["max_rss_mb"], alpha=0.5, s=10)
plt.xlabel("max_cache (MB)")
plt.ylabel("max_rss (MB)")
plt.title("max_cache vs max_rss")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/18_max_cache_vs_max_rss.png"
)
plt.close()

df["memory_overuse"] = df["max_rss_mb"] / df["memreq_mb"]
print(f"\nAverage memory overuse ratio: {df['memory_overuse'].mean():.3f}")
print(f"Median memory overuse ratio: {df['memory_overuse'].median():.3f}")
print("Memory overuse distribution:")
print(f"  < 0.5x requested: {(df['memory_overuse'] < 0.5).mean() * 100:.1f}%")
print(
    f"  0.5x - 1.0x:      {((df['memory_overuse'] >= 0.5) & (df['memory_overuse'] <= 1.0)).mean() * 100:.1f}%"
)
print(
    f"  1.0x - 2.0x:      {((df['memory_overuse'] > 1.0) & (df['memory_overuse'] <= 2.0)).mean() * 100:.1f}%"
)
print(f"  > 2.0x requested: {(df['memory_overuse'] > 2.0).mean() * 100:.1f}%")

hourly_memory = df.groupby("hour")["max_rss_mb"].mean()
daily_memory = df.groupby("day_of_week")["max_rss_mb"].mean()
print("Average memory usage by hour:")
for hour in range(24):
    if hour in hourly_memory.index:
        print(f"  Hour {hour:2d}: {hourly_memory[hour]:6.1f} MB")

print("\nAverage memory usage by day of week:")
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for day_num in range(7):
    if day_num in daily_memory.index:
        print(f"  {days[day_num]:9s}: {daily_memory[day_num]:6.1f} MB")


# Memory overuse distribution
plt.figure(figsize=(10, 6))
plt.hist(
    df["memory_overuse"],
    bins=50,
    alpha=0.7,
    color="lightcoral",
    edgecolor="black",
)
plt.axvline(x=4, color="red", linestyle="--", linewidth=2, label="Perfect Allocation")
plt.xlabel("Memory Overuse Ratio")
plt.ylabel("Frequency")
plt.title("Memory Overuse Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/19_memory_overuse.png")
plt.close()


# max_rss vs memreq with perfect allocation line (memreq = max_rss)
plt.figure(figsize=(12, 8))
plt.scatter(
    df["memreq_mb"],
    df["max_rss_mb"],
    alpha=0.5,
    s=20,
    c=df["memory_overuse"],
    cmap="RdYlBu_r",
    vmin=0,
    vmax=3,
)
plt.colorbar(label="Memory Overuse Ratio")

# Get the range for both axes
min_val = min(df["memreq_mb"].min(), df["max_rss_mb"].min())
max_val = max(df["memreq_mb"].max(), df["max_rss_mb"].max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Perfect Allocation",
)
plt.xlabel("memreq (MB)")
plt.ylabel("max_rss(MB)")
plt.title("memreq vs max_rss")
plt.xscale("log")
plt.yscale("log")

# Set the same limits for both axes
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/krebs/Distributed_Systems_Project/eda/plots/20_memreq_vs_actual.png")
plt.close()

print("\n=== JOB ANALYSIS ===")
print("Jobs distribution:")
print(df["jobs"].value_counts().sort_index())

print("\nLocal Jobs distribution:")
print(df["localJobs"].value_counts().sort_index())

jobs_stats = (
    df.groupby("jobs")
    .agg({"max_rss_mb": ["count", "mean", "std"], "localJobs": "mean"})
    .round(3)
)
jobs_stats.columns = ["count", "mean_memory", "std_memory", "avg_local_jobs"]
print("Top job configurations by frequency:")
print(jobs_stats.sort_values("count", ascending=False).head(10))

print(
    "\nMemory usage vs Jobs correlation:", df[["jobs", "max_rss_mb"]].corr().iloc[0, 1]
)
print(
    "Memory usage vs LocalJobs correlation:",
    df[["localJobs", "max_rss_mb"]].corr().iloc[0, 1],
)

print("\n=== CORRELATION ANALYSIS ===")
numeric_cols = [
    "memory_fail_count",
    "jobs",
    "localJobs",
    "max_rss",
    "max_cache",
    "memreq",
]
correlation_matrix = df[numeric_cols].corr()
print("Correlation with max_rss:")
correlations = correlation_matrix["max_rss"].sort_values(key=abs, ascending=False)
print(correlations)

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    fmt=".3f",
    cbar_kws={"shrink": 0.8},
)
plt.title("Numerical Features Correlation Matrix")
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/21_numeric_correlation_heatmap.png"
)
plt.close()

categorical_features = [
    "location",
    "branch",
    "makeType",
    "targets",
    "component",
    "architecture",
    "compiler",
    "optimization",
]

df_encoded = df.copy()
label_encoders = {}

for col in categorical_features:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# Define all predictive features for correlation analysis
predictive_features = [
    "jobs",
    "localJobs",
    "memreq",
    "location_encoded",
    "branch_encoded",
    "makeType_encoded",
    "targets_encoded",
    "component_encoded",
    "architecture_encoded",
    "compiler_encoded",
    "optimization_encoded",
]

correlation_features = predictive_features + ["max_rss"]

# Calculate correlation matrix
correlation_matrix = df_encoded[correlation_features].corr()

# Focus on correlations with target variable
target_correlations = (
    correlation_matrix["max_rss"].drop("max_rss").sort_values(key=abs, ascending=False)
)

print("\nCorrelation with max_rss (sorted by absolute value):")
for feature, corr in target_correlations.items():
    feature_name = feature.replace("_encoded", "")
    print(f"{feature_name:15s}: {corr:6.3f}")

# Plot correlation heatmap for predictive features
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix[correlation_features],
    mask=mask,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".2f",
    cbar_kws={"shrink": 0.8},
    xticklabels=[f.replace("_encoded", "") for f in correlation_features],
    yticklabels=[f.replace("_encoded", "") for f in correlation_features],
)
plt.title("Predictive Features Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/22_predictive_features_correlation.png",
)
plt.close()


print("\n=== FEATURE IMPORTANCE RANKING ===")
print("Features ranked by correlation strength with memory usage:")

for i, (feature, corr) in enumerate(target_correlations.items(), 1):
    feature_name = feature.replace("_encoded", "")
    IMPORTANCE = "High" if abs(corr) > 0.3 else "Medium" if abs(corr) > 0.1 else "Low"
    print(f"{i:2d}. {feature_name:15s}: {corr:6.3f} ({IMPORTANCE})")


# 21. Enhanced correlation heatmap for predictive features
plt.figure(figsize=(14, 12))
predictive_corr_matrix = correlation_matrix[correlation_features]
mask = np.triu(np.ones_like(predictive_corr_matrix, dtype=bool))
sns.heatmap(
    predictive_corr_matrix,
    mask=mask,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".2f",
    cbar_kws={"shrink": 0.8},
    xticklabels=[f.replace("_encoded", "") for f in correlation_features],
    yticklabels=[f.replace("_encoded", "") for f in correlation_features],
)
plt.title("Predictive Features Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/23_predictive_features_correlation.png",
    dpi=300,
)
plt.close()


# Feature importance bar plot
feature_importance = target_correlations.abs().sort_values(ascending=True).tail(10)
feature_names = [f.replace("_encoded", "") for f in feature_importance.index]
plt.figure(figsize=(14, 12))
plt.barh(range(len(feature_importance)), feature_importance.values)
plt.yticks(range(len(feature_importance)), feature_names)
plt.xlabel("Absolute Correlation with max_rss")
plt.title("Top 10 Most Predictive Features")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/home/krebs/Distributed_Systems_Project/eda/plots/24_detailed_analysis.png"
)
plt.close()
