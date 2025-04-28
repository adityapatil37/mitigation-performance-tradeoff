# Advanced Security Mitigation Analyzer

A Python-based benchmarking framework to evaluate the performance and memory overhead of common exploit mitigation techniques—ASLR, DEP, and CFI—across different environment profiles.
<br><br>
This tool provides a systematic framework for evaluating the performance impact of modern security mitigations (ASLR, DEP, CFI) across heterogeneous computing environments. Designed for cybersecurity professionals, system architects, and DevOps teams, it enables quantitative analysis of security-performance tradeoffs through statistically rigorous benchmarking. The solution addresses critical industry needs for data-driven security configuration decisions in contexts ranging from embedded systems to cloud infrastructure.

---

## Table of Contents

1. [Features](#features)
2. [Environment Profiles](#environment-profiles)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Script Overview](#script-overview)
7. [Output and Results](#output-and-results)
8. [Extending and Customization](#extending-and-customization)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Features

- **Simulate ASLR, DEP, and CFI** with configurable parameters.
- **Benchmark** execution time and memory usage with warmup and statistical analysis.
- **Compare** multiple mitigation configurations with p‑values, effect sizes, and mean differences.
- **Plot** performance profiles and save as high-resolution PNG(s).
- **Save** full results and comparisons to JSON for downstream analysis.

---

## Environment Profiles

Predefined profiles optimize parameters for various target platforms:

| Profile             | ASLR Iterations | DEP Checks | CFI Depth | Memory Size | Parallel | Notes               |
|---------------------|-----------------|------------|-----------|-------------|----------|---------------------|
| embedded            | 50              | 10         | 2         | 1 KB        | No       | Constrained devices |
| server              | 1000            | 100        | 5         | 1 MB        | Yes      | High-throughput     |
| desktop             | 500             | 50         | 3         | 64 KB       | No       | Balanced            |
| cloud_serverless    | 2000            | 150        | 4         | 128 KB      | Yes      | Ephemeral, cold starts |
| …                   | …               | …          | …         | …           | …        | …                   |

Profiles can be found and modified in the `ENV_PROFILES` dictionary.

---

## Dependencies

- Python **3.7+**
- `psutil`
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies via pip:

```bash
pip install psutil numpy scipy matplotlib
```

---

## Installation

1. Clone the repository or download the `mitigation_benchmark.py` script.
2. Ensure dependencies are installed (see [Dependencies](#dependencies)).
3. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

---

## Usage

Run the benchmarking script with command-line options:

```bash
python mitigation_benchmark.py \
  --environment <profile> \
  --iterations <num> \
  --confidence <level> \
  [--plot]
```

- `--environment`, `-e`: Target profile (default: `desktop`). Options include: `embedded`, `server`, `desktop`, `cloud_serverless`, `iot_industrial`, `automotive`, `gaming_console`, `medical_device`, etc.
- `--iterations`, `-i`: Number of measurement iterations per configuration (default: `50`).
- `--confidence`, `-c`: Confidence level for statistical intervals (default: `0.95`).
- `--plot`: Generate and save a performance comparison plot (`analysis_<env>.png`).

**Example:**

```bash
python mitigation_benchmark.py -e server -i 100 -c 0.99 --plot
```

---

## Script Overview

1. **Simulate Mitigations**:
   - **ASLR**: Random memory-mapping reads.
   - **DEP**: Read/write checks on a memory buffer.
   - **CFI**: Call-stack validation with optional parallel execution.

2. **Workload Creation**:
   - Base computational task sums large arrays.
   - Decorators apply selected mitigations.

3. **Benchmarking**:
   - Warmup runs to stabilize environment.
   - Measure execution time (`time.perf_counter`) and memory (`psutil`).
   - Collect samples and compute robust statistics (mean, stdev, confidence intervals).

4. **Comparison**:
   - Pairwise statistical tests (Welch’s t-test).
   - Calculate effect sizes and significance.

5. **Reporting**:
   - Print summary to console.
   - Save detailed JSON results (`results_<env>.json`).
   - (Optional) Plot with error bars for time and memory.

---

## Output and Results

- **Console**: Summary of each configuration, including mean ± stdev and 95% CI.
- **JSON**: `results_<environment>.json` contains:
  - Timestamp and environment name.
  - Per-configuration samples and statistics.
  - Pairwise comparison results (p-values, effect sizes).
- **Plot**: `analysis_<environment>.png` (if `--plot` is provided).

---

## Extending and Customization

- **Add Profiles**: Extend `ENV_PROFILES` with new keys and parameter sets.
- **New Mitigations**: Implement additional decorators following the existing patterns.
- **Workload Tweaks**: Modify `create_workload` to change computational tasks.
- **Reporting**: Integrate other visualizations or export formats.

---

## Troubleshooting

- **High Memory Usage**: Reduce `memory_size` in profile or lower iterations.
- **Permissions Errors**: On some OSes, memory mapping may require elevated privileges.
- **Missing Modules**: Ensure all dependencies are installed in the active Python environment.

---

## License

Apache 2.0 with Commons Clause - See LICENSE for details.

Commercial Use
Contact adityapa37@gmail.com for enterprise licensing options.
