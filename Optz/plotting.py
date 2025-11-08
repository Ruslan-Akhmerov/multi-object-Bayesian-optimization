import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from .gaussian_process import update_gaussian_process  # Add this import
from scipy.stats import norm

def save_data_to_txt(filename, **kwargs):
    """Helper function to save data to text files"""
    with open(filename, 'w') as f:
        for key, values in kwargs.items():
            f.write(f"{key}\n")
            if isinstance(values, (list, np.ndarray)):
                np.savetxt(f, values, fmt='%f')
            else:
                f.write(f"{values}\n")
            f.write("\n")

def plot_iteration(optimizer, iteration, x_range):
    """Create and save iteration plots"""
    if len(optimizer.u_history) < 2:
        print("Skipping visualization - not enough data")
        return
        
    plt.figure(figsize=(16, 12))
    base_filename = f'iterations/iteration_{iteration:02d}'
    
    # 1. Optimization history
    plt.subplot(2, 2, 1)
    plt.plot(optimizer.u_history, optimizer.gap_history, 'bo-', label='Calculated values')
    plt.axhline(optimizer.target_gap, color='r', linestyle='--', 
                label='Target value')
    plt.xlabel('Hubbard parameter U (eV)', fontsize=12)
    plt.ylabel('Band gap (eV)', fontsize=12)
    plt.title(f'Optimization history (Iteration {iteration})', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Save optimization history data
    save_data_to_txt(
        f'{base_filename}_optimization_history.txt',
        u_values=optimizer.u_history,
        gap_values=optimizer.gap_history,
        target_gap=optimizer.target_gap
    )
    
    # 2. Convergence
    plt.subplot(2, 2, 2)
    deviations = np.abs(np.array(optimizer.gap_history) - optimizer.target_gap)
    plt.plot(range(len(deviations)), deviations, 'ro-')
    plt.xlabel('Iteration number', fontsize=12)
    plt.ylabel('Deviation from target (eV)', fontsize=12)
    plt.title('Convergence', fontsize=14)
    plt.grid(True)
    
    # Save convergence data
    save_data_to_txt(
        f'{base_filename}_convergence.txt',
        iteration_numbers=list(range(len(deviations))),
        deviations=deviations
    )
    
    # 3. Gaussian process with real data approximation
    plt.subplot(2, 2, 3)
    X_train = np.array(optimizer.u_history)
    y_train = np.array([-((gap - optimizer.target_gap)**2) for gap in optimizer.gap_history])
    
    # Sort data for approximation
    sort_idx = np.argsort(optimizer.u_history)
    u_sorted = np.array(optimizer.u_history)[sort_idx]
    y_sorted = np.array(y_train)[sort_idx]
    
    # Cubic spline approximation
    cs = CubicSpline(u_sorted, y_sorted)
    y_interp = cs(x_range)
    
    # Gaussian process (recalculate here!)
    mu, cov = update_gaussian_process(X_train, y_train, x_range)
    std_dev = np.sqrt(np.diag(cov))
    
    # Real data approximation
    plt.plot(x_range, y_interp, 'r-', label='Point approximation')
    plt.scatter(optimizer.u_history, y_train, c='blue', s=50, label='Observations')
    plt.plot(x_range, mu, 'k-', label='Gaussian process')
    plt.fill_between(x_range, 
                     mu - 1.96*std_dev, 
                     mu + 1.96*std_dev, 
                     alpha=0.2, color='gray', label='95% Confidence interval')
    plt.xlabel('Hubbard parameter U (eV)', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    plt.title('Training', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Save Gaussian process data
    save_data_to_txt(
        f'{base_filename}_gaussian_process.txt',
        x_range=x_range,
        y_interpolated=y_interp,
        u_history=optimizer.u_history,
        y_train=y_train,
        mu=mu,
        std_dev=std_dev
    )
    
    # 4. NEW: Expected Improvement acquisition function
    
    plt.subplot(2, 2, 4)

    # Calculate EI
    best_y = np.max(y_train)  # Since we're maximizing the negative squared error
    #xi = np.clip(0.3 * (1 - (iteration/30)**2), 0.01, 0.3)  # Exploration-exploitation parameter
    xi = 0.01
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (mu - best_y - xi) / std_dev
        Z[std_dev == 0] = 0  # Handle division by zero
    
    EI = (mu - best_y - xi) * norm.cdf(Z) + std_dev * norm.pdf(Z)
    EI[std_dev == 0] = 0
    
    next_idx = np.argmax(EI)
    next_point = x_range[next_idx]
    
    plt.plot(x_range, EI, 'g-', label='Expected Improvement (ξ=0.01)')
    plt.scatter(next_point, EI[next_idx], 
               marker='*', color='red', s=200, label='Best point')
    plt.xlabel('Hubbard parameter U (eV)', fontsize=12)
    plt.ylabel('Acquisition Value', fontsize=12)
    plt.title('Acquisition Function (EI)', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Save acquisition function data
    save_data_to_txt(
        f'{base_filename}_acquisition_function.txt',
        x_range=x_range,
        EI_values=EI,
        next_point=next_point,
        next_point_value=EI[next_idx]
    )
    
    plt.tight_layout()
    
    """
    

    # 4. Acquisition function (UCB)
    plt.subplot(2, 2, 4)
    kappa = 1.96  # 99% confidence interval
    ucb = mu + kappa * std_dev
    next_idx = np.argmax(ucb)
    next_point = x_range[next_idx]
    print(next_point)
    
    plt.plot(x_range, ucb, 'g-', label='UCB (kappa=2.576)')
    plt.scatter(next_point, ucb[next_idx], 
           marker='*', color='red', s=200, label='Best point')
    plt.xlabel('Hubbard parameter U (eV)', fontsize=12)
    plt.ylabel('Acquisition', fontsize=12)
    plt.title('Point selection strategy', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    """
    
    # Create iterations directory if it doesn't exist
    os.makedirs('iterations', exist_ok=True)
    plot_file = f'iterations/iteration_{iteration:02d}_full.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Iteration plot saved as {plot_file}")

def plot_final_results(optimizer):
    """Plot final results and save data to text files"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save all history data
    save_data_to_txt(
        'results/final_optimization_history.txt',
        u_values=optimizer.u_history,
        gap_values=optimizer.gap_history,
        target_gap=optimizer.target_gap
    )
    
    # Calculate and save deviations
    deviations = np.abs(np.array(optimizer.gap_history) - optimizer.target_gap)
    save_data_to_txt(
        'results/final_convergence.txt',
        iteration_numbers=list(range(len(deviations))),
        deviations=deviations
    )
    
    # [Rest of your original plot_final_results implementation]
    # Make sure to save any additional data you plot to text files
    print("Final results data saved to text files in 'results' directory")
