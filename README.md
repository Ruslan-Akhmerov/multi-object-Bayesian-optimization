# Hubbard Parameter Optimizer

A Python package for automated optimization of Hubbard U parameters in Quantum ESPRESSO calculations using Bayesian optimization.

## Description

This tool automatically finds the optimal Hubbard U parameter that reproduces experimental band gaps by:

- Running sequential Quantum ESPRESSO calculations (SCF → NSCF → Bands)
- Analyzing band structure results
- Using Gaussian Process regression to suggest next U values
- Continuing until target band gap precision is achieved

## Features

- **Bayesian Optimization** with Expected Improvement acquisition function
- **Automated workflow** for Quantum ESPRESSO calculations
- **Comprehensive logging** of all iterations
- **Visualization** of optimization progress
- **Error handling** and recovery mechanisms

## Quick Start

1. Install dependencies:
```bash
pip install numpy scipy matplotlib
