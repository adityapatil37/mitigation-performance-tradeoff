import time
import functools
import statistics
import argparse
import matplotlib.pyplot as plt
import ctypes
import mmap
import os
import random
import gc
import json
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import psutil
from scipy.stats import ttest_ind, t
import numpy as np
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Environment profiles configuration
ENV_PROFILES = {
    'embedded': {
        'aslr_iterations': 50,
        'dep_checks': 10,
        'cfi_depth': 2,
        'memory_size': 1024,
        'parallel': False
    },
    'server': {
        'aslr_iterations': 1000,
        'dep_checks': 100,
        'cfi_depth': 5,
        'memory_size': 1048576,
        'parallel': True
    },
    'desktop': {
        'aslr_iterations': 500,
        'dep_checks': 50,
        'cfi_depth': 3,
        'memory_size': 65536,
        'parallel': False
    },
    'cloud_serverless': {
        'aslr_iterations': 2000,   # Frequent cold starts
        'dep_checks': 150,
        'cfi_depth': 4,
        'memory_size': 131072,     # 128KB
        'parallel': True,
        'ephemeral': True
    },
    'iot_industrial': {
        'aslr_iterations': 100,    # Constrained devices
        'dep_checks': 75,
        'cfi_depth': 3,
        'memory_size': 512,        # 0.5KB
        'parallel': False,
        'real_time': True
    },
    'automotive': {
        'aslr_iterations': 300,    # Balance of safety/performance
        'dep_checks': 200,
        'cfi_depth': 4,
        'memory_size': 32768,      # 32KB
        'parallel': False,
        'safety_critical': True
    },
    'gaming_console': {
        'aslr_iterations': 1500,   # High-performance security
        'dep_checks': 200,
        'cfi_depth': 6,
        'memory_size': 2097152,    # 2MB
        'parallel': True,
        'high_perf': True
    },
    'medical_device': {
        'aslr_iterations': 400,    # Regulatory compliance focus
        'dep_checks': 300,
        'cfi_depth': 5,
        'memory_size': 16384,      # 16KB
        'parallel': False,
        'fda_class': 'II'
    },
    'edge_computing': {
        'aslr_iterations': 800,    # Distributed systems balance
        'dep_checks': 120,
        'cfi_depth': 4,
        'memory_size': 262144,     # 256KB
        'parallel': True,
        'low_latency': True
    },
    'mobile': {
        'aslr_iterations': 600,    # Battery-constrained security
        'dep_checks': 180,
        'cfi_depth': 4,
        'memory_size': 524288,     # 512KB
        'parallel': True,
        'power_efficient': True
    },
    'quantum': {
        'aslr_iterations': 5000,   # Post-quantum preparation
        'dep_checks': 500,
        'cfi_depth': 8,
        'memory_size': 4194304,    # 4MB
        'parallel': True,
        'quantum_safe': True
    },
    'smart_contract': {
        'aslr_iterations': 200,    # Blockchain-specific needs
        'dep_checks': 400,
        'cfi_depth': 7,
        'memory_size': 8192,       # 8KB
        'parallel': False,
        'immutable': True
    },
    'military': {
        'aslr_iterations': 3000,   # Extreme security posture
        'dep_checks': 1000,
        'cfi_depth': 9,
        'memory_size': 1048576,    # 1MB
        'parallel': True,
        'air_gapped': True
    },
    'blockchain': {
        'aslr_iterations': 1000,   # Decentralized security
        'dep_checks': 300,
        'cfi_depth': 6,
        'memory_size': 131072,     # 128KB
        'parallel': True,
        'distributed': True
    },
    '5g_network': {
        'aslr_iterations': 800,    # High-speed network security
        'dep_checks': 250,
        'cfi_depth': 5,
        'memory_size': 524288,     # 512KB
        'parallel': True,
        'low_latency': True
    },
    'quantum_computing': {
        'aslr_iterations': 10000,  # Quantum-specific needs
        'dep_checks': 2000,
        'cfi_depth': 10,
        'memory_size': 8388608,    # 8MB
        'parallel': True,
        'quantum_safe': True
    },
    'edge_ai': {
        'aslr_iterations': 1200,   # AI model security
        'dep_checks': 300,
        'cfi_depth': 6,
        'memory_size': 262144,     # 256KB
        'parallel': True,
        'ai_optimized': True
    },
    'robotics': {
        'aslr_iterations': 700,    # Real-time robotics security
        'dep_checks': 200,
        'cfi_depth': 5,
        'memory_size': 131072,     # 128KB
        'parallel': False,
        'real_time': True
    },
    'smart_home': {
        'aslr_iterations': 300,    # Home automation security
        'dep_checks': 150,
        'cfi_depth': 4,
        'memory_size': 65536,      # 64KB
        'parallel': False,
        'user_friendly': True
    },
    'augmented_reality': {
        'aslr_iterations': 900,    # AR-specific needs
        'dep_checks': 250,
        'cfi_depth': 5,
        'memory_size': 524288,     # 512KB
        'parallel': True,
        'immersive': True
    },
    'virtual_reality': {
        'aslr_iterations': 1200,   # VR-specific needs
        'dep_checks': 300,
        'cfi_depth': 6,
        'memory_size': 1048576,    # 1MB
        'parallel': True,
        'immersive': True
    },
    'drones': {
        'aslr_iterations': 600,    # Drone-specific needs
        'dep_checks': 200,
        'cfi_depth': 5,
        'memory_size': 131072,     # 128KB
        'parallel': False,
        'real_time': True
    },
    'smart_grid': {
        'aslr_iterations': 800,    # Smart grid security
        'dep_checks': 250,
        'cfi_depth': 6,
        'memory_size': 262144,     # 256KB
        'parallel': True,
        'low_latency': True
    },
    'biometric': {
        'aslr_iterations': 400,    # Biometric data security
        'dep_checks': 150,
        'cfi_depth': 4,
        'memory_size': 65536,      # 64KB
        'parallel': False,
        'privacy_sensitive': True
    },
    'wearable_tech': {
        'aslr_iterations': 500,    # Wearable device security
        'dep_checks': 200,
        'cfi_depth': 5,
        'memory_size': 32768,      # 32KB
        'parallel': False,
        'battery_constrained': True
    },
    'supply_chain': {
        'aslr_iterations': 1000,   # Supply chain security
        'dep_checks': 300,
        'cfi_depth': 6,
        'memory_size': 131072,     # 128KB
        'parallel': True,
        'immutable': True
    },
    'automated_testing': {
        'aslr_iterations': 200,    # Automated testing security
        'dep_checks': 100,
        'cfi_depth': 3,
        'memory_size': 8192,       # 8KB
        'parallel': False,
        'test_automation': True
    },
    'data_center': {
        'aslr_iterations': 1500,   # Data center security
        'dep_checks': 500,
        'cfi_depth': 7,
        'memory_size': 2097152,    # 2MB
        'parallel': True,
        'high_perf': True
    },
    'telecommunications': {
        'aslr_iterations': 1200,   # Telecom security
        'dep_checks': 400,
        'cfi_depth': 6,
        'memory_size': 524288,     # 512KB
        'parallel': True,
        'low_latency': True
    },
    'financial_services': {
        'aslr_iterations': 800,    # Financial security
        'dep_checks': 300,
        'cfi_depth': 5,
        'memory_size': 262144,     # 256KB
        'parallel': True,
        'privacy_sensitive': True
    },
    'e_commerce': {
        'aslr_iterations': 600,    # E-commerce security
        'dep_checks': 200,
        'cfi_depth': 4,
        'memory_size': 131072,     # 128KB
        'parallel': False,
        'user_friendly': True
    },
    'social_media': {
        'aslr_iterations': 500,    # Social media security
        'dep_checks': 150,
        'cfi_depth': 3,
        'memory_size': 65536,      # 64KB
        'parallel': False,
        'user_friendly': True
    },
    'cloud_storage': {
        'aslr_iterations': 1000,   # Cloud storage security
        'dep_checks': 300,
        'cfi_depth': 6,
        'memory_size': 1048576,    # 1MB
        'parallel': True,
        'privacy_sensitive': True
    },
    'content_delivery': {
        'aslr_iterations': 1200,   # Content delivery security
        'dep_checks': 400,
        'cfi_depth': 5,
        'memory_size': 524288,     # 512KB
        'parallel': True,
        'low_latency': True
    },
    'gaming': {
        'aslr_iterations': 1500,   # Gaming security
        'dep_checks': 500,
        'cfi_depth': 7,
        'memory_size': 2097152,    # 2MB
        'parallel': True,
        'high_perf': True
    },
}

current_env = 'desktop'
EPSILON = 1e-9  # Minimum value for standard deviation calculations

# Mitigation simulation decorators
def simulate_aslr(env_profile):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            iterations = env_profile['aslr_iterations']
            mem_size = env_profile['memory_size']

            if os.name == 'nt':
                buffer = mmap.mmap(-1, mem_size, access=mmap.ACCESS_READ)
            else:
                buffer = mmap.mmap(-1, mem_size, prot=mmap.PROT_READ)
            
            for _ in range(iterations):
                buffer.seek(random.randint(0, mem_size-1))
                buffer.read(1)
            buffer.close()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def simulate_dep(env_profile):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            checks = env_profile['dep_checks']
            mem_size = env_profile['memory_size']

            try:
                if os.name == 'nt':
                    buf = mmap.mmap(-1, mem_size, access=mmap.ACCESS_WRITE)
                else:
                    buf = mmap.mmap(-1, mem_size, prot=mmap.PROT_READ|mmap.PROT_WRITE)
                
                for _ in range(checks):
                    buf.write(b'\x90' * 10)
                    buf.seek(0)
                    buf.read(10)
                
                buf.close()
            except Exception as e:
                pass
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class CFIValidator:
    def __init__(self, env_profile):
        self.allowed_paths = set()
        self.depth = env_profile['cfi_depth']
        self.history = []
        
    def validate(self, func):
        current_frame = func.__name__
        if len(self.history) >= self.depth:
            path = tuple(self.history[-self.depth:])
            if path not in self.allowed_paths:
                self.allowed_paths.add(path)
        
        self.history.append(current_frame)
        if len(self.history) > self.depth:
            self.history.pop(0)

def simulate_cfi(env_profile):
    validator = CFIValidator(env_profile)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            validator.validate(func)
            
            if env_profile['parallel']:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result()
            else:
                result = func(*args, **kwargs)
            
            end = time.perf_counter()
            validation_time = (end - start) * 0.1
            time.sleep(validation_time)
            
            return result
        return wrapper
    return decorator

# Workload creation
def create_workload(env_profile, mitigations):
    def base_computation():
        size = env_profile['memory_size']
        array = list(range(size))
        return sum(array) + sum(array[::random.randint(1, 10)])
    
    func = base_computation
    if 'aslr' in mitigations:
        func = simulate_aslr(env_profile)(func)
    if 'dep' in mitigations:
        func = simulate_dep(env_profile)(func)
    if 'cfi' in mitigations:
        func = simulate_cfi(env_profile)(func)
    
    return func

# Enhanced benchmarking system
def benchmark(func, iterations=20, warmup=5, confidence=0.95):
    # --- Validate inputs ---
    try:
        iterations = int(iterations)
        warmup     = int(warmup)
        confidence = float(confidence)
        if iterations < 1 or warmup < 0 or not (0.0 < confidence < 1.0):
            raise ValueError
    except ValueError:
        raise ValueError("`iterations` ≥1 (int), `warmup` ≥0 (int), and `confidence` ∈ (0,1)")

    # --- Acquire process handle safely ---
    try:
        process = psutil.Process(os.getpid())
    except Exception as e:
        logging.error(f"Failed to access process info: {e}")
        # Return empty stats in case of failure
        return {
            'time':   {'samples': [], 'stats': {}},
            'memory': {'samples': [], 'stats': {}}
        }

    results = {
        'time':   {'samples': [], 'stats': {}},
        'memory': {'samples': [], 'stats': {}}
    }

    # --- Warmup phase ---
    for i in range(1, warmup + 1):
        gc.collect()
        try:
            func()
        except MemoryError as me:
            logging.warning(f"Warmup #{i} out of memory: {me}")
        except Exception as e:
            logging.warning(f"Warmup #{i} failed: {e}")

    # --- Measurement phase ---
    valid_runs = 0
    attempts   = 0
    max_attempts = iterations * 3

    while valid_runs < iterations and attempts < max_attempts:
        attempts += 1
        gc.collect()

        # snapshot memory before
        try:
            mem_before = process.memory_info().rss
        except Exception as e:
            logging.error(f"Could not read memory before run: {e}")
            break

        try:
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start

            # snapshot memory after
            mem_after = process.memory_info().rss

            results['time']['samples'].append(elapsed)
            results['memory']['samples'].append(mem_after - mem_before)
            valid_runs += 1

        except MemoryError as me:
            logging.error(f"Run #{attempts} out of memory: {me}")
        except Exception as e:
            logging.error(f"Run #{attempts} failed: {e}")

    if valid_runs < iterations:
        logging.warning(f"Completed only {valid_runs}/{iterations} successful runs after {attempts} attempts")

    # --- Robust stats helper ---
    def safe_stats(data):
        if not data:
            return {
                'mean':   0.0, 'stdev': 0.0,
                'ci_low': 0.0, 'ci_high': 0.0,
                'median': 0.0, 'min':    0.0,
                'max':    0.0, 'iqr':    []
            }
        mean = statistics.mean(data)
        stdev = statistics.stdev(data) if len(data) > 1 else 0.0
        try:
            ci_low, ci_high = t.interval(
                confidence,
                len(data) - 1,
                loc=mean,
                scale=stdev / math.sqrt(len(data)) if stdev > EPSILON else EPSILON
            )
        except Exception:
            ci_low, ci_high = mean, mean

        return {
            'mean':    mean,
            'stdev':   stdev,
            'ci_low':  ci_low if not math.isnan(ci_low) else mean,
            'ci_high': ci_high if not math.isnan(ci_high) else mean,
            'median':  statistics.median(data),
            'min':     min(data),
            'max':     max(data),
            'iqr':     statistics.quantiles(data, n=4) if len(data) >= 4 else []
        }

    # --- Compute final statistics ---
    results['time']['stats']   = safe_stats(results['time']['samples'])
    results['memory']['stats'] = safe_stats(results['memory']['samples'])

    logging.info(f"Benchmark complete: {valid_runs} successful runs")
    return results

def compare_configurations(results):
    """Robust statistical comparison with error protection"""
    comparisons = []
    metrics = ['time', 'memory']
    
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            comparison = {
                'config_a': results[i]['mitigations'],
                'config_b': results[j]['mitigations']
            }
            
            for metric in metrics:
                a_data = results[i]['results'][metric]['samples']
                b_data = results[j]['results'][metric]['samples']
                
                # Calculate means and standard deviations
                a_mean = statistics.mean(a_data) if a_data else 0.0
                b_mean = statistics.mean(b_data) if b_data else 0.0
                a_std = statistics.stdev(a_data) if len(a_data) > 1 else 0.0
                b_std = statistics.stdev(b_data) if len(b_data) > 1 else 0.0
                n_a, n_b = len(a_data), len(b_data)
                
                # Calculate pooled standard deviation
                if (n_a + n_b - 2) > 0:
                    pooled_var = ((n_a-1)*a_std**2 + (n_b-1)*b_std**2) / (n_a + n_b - 2)
                    pooled_std = math.sqrt(pooled_var)
                else:
                    pooled_std = 0.0
                
                # Calculate effect size with epsilon protection
                denominator = pooled_std if pooled_std > EPSILON else EPSILON
                effect_size = (b_mean - a_mean) / denominator
                
                # Perform t-test with fallback
                try:
                    t_stat, p_value = ttest_ind(a_data, b_data, equal_var=False)
                except:
                    p_value = 1.0
                
                comparison[metric] = {
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'mean_diff': b_mean - a_mean,
                    'pooled_std': pooled_std
                }
            
            comparisons.append(comparison)
    
    return comparisons

def plot_comparison(results, env_name):
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Time performance
    time_means = [c['results']['time']['stats']['mean'] for c in results]
    time_ci = [(c['results']['time']['stats']['ci_high'] - c['results']['time']['stats']['ci_low'])/2 
               for c in results]
    labels = ['+'.join(c['mitigations']) or 'Baseline' for c in results]
    
    ax1.bar(labels, time_means, yerr=time_ci, capsize=5, alpha=0.6, color='b')
    ax1.set_ylabel('Execution Time (s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Memory usage
    ax2 = ax1.twinx()
    mem_means = [c['results']['memory']['stats']['mean']/(1024*1024) for c in results]
    mem_ci = [(c['results']['memory']['stats']['ci_high'] - c['results']['memory']['stats']['ci_low'])/2 
              for c in results]
    
    ax2.errorbar(labels, mem_means, yerr=mem_ci, fmt='r-o', markersize=8, 
                capsize=5, linewidth=2)
    ax2.set_ylabel('Memory Usage (MB)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title(f'Performance Profile: {env_name.capitalize()}')
    fig.tight_layout()
    plt.savefig(f'analysis_{env_name}.png', dpi=300)
    plt.close()

def main(args):
    global current_env
    current_env = args.environment
    env_profile = ENV_PROFILES[current_env]
    
    test_configs = [
        {'mitigations': []},
        {'mitigations': ['aslr']},
        {'mitigations': ['dep']},
        {'mitigations': ['cfi']},
        {'mitigations': ['aslr', 'dep']},
        {'mitigations': ['aslr', 'cfi']},
        {'mitigations': ['dep', 'cfi']},
        {'mitigations': ['aslr', 'dep', 'cfi']}
    ]
    
    print(f"Benchmarking in {current_env} environment ({args.iterations} iterations)")
    results = []
    
    for config in test_configs:
        workload = create_workload(env_profile, config['mitigations'])
        print(f"\nTesting {'+'.join(config['mitigations']) or 'Baseline'}...")
        
        bench_results = benchmark(
            workload,
            iterations=args.iterations,
            confidence=args.confidence
        )
        config['results'] = bench_results
        results.append(config)
        
        # Print results
        t_stats = bench_results['time']['stats']
        m_stats = bench_results['memory']['stats']
        print(f"  Time: {t_stats['mean']:.4f}s ± {t_stats['stdev']:.4f}")
        print(f"    95% CI: [{t_stats['ci_low']:.4f}, {t_stats['ci_high']:.4f}]")
        print(f"  Memory: {m_stats['mean']/1024:.2f}KB ± {m_stats['stdev']/1024:.2f}KB")
    
    # Statistical comparisons
    comparisons = compare_configurations(results)
    print("\n\nStatistical Significance:")
    for comp in comparisons:
        print(f"\n{comp['config_a']} vs {comp['config_b']}:")
        for metric in ['time', 'memory']:
            stats = comp[metric]
            sig = ("***" if stats['p_value'] < 0.001 else
                   "**" if stats['p_value'] < 0.01 else
                   "*" if stats['p_value'] < 0.05 else "ns")
            print(f"  {metric.capitalize()}:")
            print(f"    p-value: {stats['p_value']:.4e} {sig}")
            print(f"    Effect size: {stats['effect_size']:.2f} ({'small' if abs(stats['effect_size']) < 0.5 else 'medium' if abs(stats['effect_size']) < 0.8 else 'large'})")
            print(f"    Mean difference: {stats['mean_diff']:.4f}")
            print(f"    Pooled std: {stats['pooled_std']:.4f}")

    if args.plot:
        plot_comparison(results, current_env)
        print(f"\nPlot saved to analysis_{current_env}.png")
    
    # Save results
    with open(f'results_{current_env}.json', 'w') as f:
        json.dump({
            'environment': current_env,
            'timestamp': datetime.now().isoformat(),
            'configurations': results,
            'comparisons': comparisons
        }, f, indent=2)
    
    print(f"\nFull results saved to results_{current_env}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Security Mitigation Analyzer")
    parser.add_argument('-e', '--environment', choices=['embedded', 'server', 'desktop', 'cloud_serverless',
                       'iot_industrial', 'automotive', 'gaming_console', 'medical_device',],
                       default='desktop', help="Target environment profile")
    parser.add_argument('-i', '--iterations', type=int, default=50,
                       help="Number of benchmark iterations")
    parser.add_argument('-c', '--confidence', type=float, default=0.95,
                       help="Confidence level for statistical intervals")
    parser.add_argument('--plot', action='store_true',
                       help="Generate performance comparison plot")
    
    args = parser.parse_args()
    main(args)