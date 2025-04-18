import time
import functools
import statistics
import argparse
import matplotlib.pyplot as plt
import ctypes
import mmap
import os
import random
from concurrent.futures import ThreadPoolExecutor

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
    }
}

# Global configuration
current_env = 'desktop'


def simulate_aslr(env_profile):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            iterations = env_profile['aslr_iterations']
            mem_size = env_profile['memory_size']

            # Platform-agnostic memory mapping
            if os.name == 'nt':  # Windows
                buffer = mmap.mmap(-1, mem_size, access=mmap.ACCESS_READ)
            else:  # Unix-like systems
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
    """Control Flow Integrity validator with environment-aware policy"""
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
    """Simulate CFI with call graph validation"""
    validator = CFIValidator(env_profile)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            validator.validate(func)
            
            # Environment-aware validation frequency
            if env_profile['parallel']:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result()
            else:
                result = func(*args, **kwargs)
            
            end = time.perf_counter()
            validation_time = (end - start) * 0.1  # 10% of runtime
            time.sleep(validation_time)
            
            return result
        return wrapper
    return decorator

# -----------------------------------------------------------------------------------
# Workload Functions with Configurable Mitigations
# -----------------------------------------------------------------------------------

def create_workload(env_profile, mitigations):
    """Factory function to create workload with specified mitigations"""
    def base_computation():
        # CPU-bound task with memory access patterns
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

# -----------------------------------------------------------------------------------
# Enhanced Benchmarking System
# -----------------------------------------------------------------------------------

def benchmark(func, iterations=10, warmup=3):
    """Advanced benchmarking with warmup cycles and statistical analysis"""
    # Warmup phase
    for _ in range(warmup):
        func()
    
    # Measurement phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'avg': statistics.mean(times),
        'stdev': statistics.stdev(times),
        'min': min(times),
        'max': max(times),
        'times': times
    }


def plot_comparison(results, env_name):
    """Generate comparative visualization for environment profile"""
    labels = []
    avg_times = []
    std_devs = []
    
    for config in results:
        label = '+'.join(config['mitigations']) if config['mitigations'] else 'Baseline'
        labels.append(label)
        avg_times.append(config['results']['avg'])
        std_devs.append(config['results']['stdev'])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, avg_times, yerr=std_devs, capsize=5)
    plt.xlabel('Mitigation Configuration')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Comparison: {env_name.capitalize()} Environment')
    plt.xticks(rotation=45, ha='right')
    
    # Add performance annotations
    for bar, avg, std in zip(bars, avg_times, std_devs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{avg:.4f} ± {std:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'performance_{env_name}.png')
    plt.close()

def main(args):
    global current_env
    current_env = args.environment
    env_profile = ENV_PROFILES[current_env]
    
    print(f"Selected Environment: {current_env}")
    print(f"Iterations: {args.iterations}")
    
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
    
    results = []
    
    print(f"Testing in {current_env} environment...\n")
    for config in test_configs:
        workload = create_workload(env_profile, config['mitigations'])
        print(f"Benchmarking {'+'.join(config['mitigations']) or 'Baseline'}...")
        bench_results = benchmark(workload, args.iterations)
        config['results'] = bench_results
        results.append(config)
        
        print(f"  Average: {bench_results['avg']:.6f}s")
        print(f"  Std Dev: {bench_results['stdev']:.6f}s")
        print(f"  Range: {bench_results['min']:.6f}s - {bench_results['max']:.6f}s\n")
    
    # Only generate plot if --plot is specified
    if args.plot:
        plot_comparison(results, current_env)
        print(f"Plot saved to performance_{current_env}.png")
    
    with open(f'results_{current_env}.txt', 'w') as f:
        for config in results:
            f.write(f"Configuration: {'+'.join(config['mitigations']) or 'Baseline'}\n")
            f.write(f"  Average: {config['results']['avg']:.6f}s\n")
            f.write(f"  Std Dev: {config['results']['stdev']:.6f}s\n")
            f.write(f"  Range: {config['results']['min']:.6f}s - {config['results']['max']:.6f}s\n\n")
    
    print(f"\nResults saved to results_{current_env}.txt")

# -----------------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Mitigation Overhead Analyzer")
    parser.add_argument('-e', '--environment', choices=['embedded', 'server', 'desktop'],
                       default='desktop', help="Target environment profile")
    parser.add_argument('-i', '--iterations', type=int, default=20,
                       help="Number of benchmark iterations")
    parser.add_argument('--plot', action='store_true',
                       help="Generate performance comparison plot")
    args = parser.parse_args()
    
    main(args)