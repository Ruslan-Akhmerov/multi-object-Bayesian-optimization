import numpy as np
import os
import subprocess
from datetime import datetime
from .gaussian_process import update_gaussian_process
from .plotting import plot_iteration, plot_final_results
from .file_utils import save_iteration_files, save_to_npz
from scipy.stats import norm

class HubbardOptimizer:
    def __init__(self, compound="CoO", target_gap=2.6, initial_u=6.3, 
                 bounds=(0, 10), max_iter=20, precision=0.01):
        self.compound = compound
        self.target_gap = target_gap
        
        if isinstance(initial_u, (list, np.ndarray)):
            self.initial_u = initial_u
        else:
            self.initial_u = [initial_u]
        
        self.current_u = self.initial_u[0]
        self.bounds = bounds
        self.max_iter = max_iter
        self.precision = precision
        self.u_history = []
        self.gap_history = []
        self.files = [f'{compound}.scf.in', f'{compound}.nscf.in', f'{compound}.band.in']
        self.start_time = datetime.now()
        
        print("\n" + "="*60)
        print("Initializing Hubbard parameter optimizer")
        print(f"Compound: {compound}")
        print(f"Target band gap: {target_gap} eV")
        print(f"Initial U value: {initial_u} eV")
        print(f"Search bounds: {bounds}")
        print(f"Max iterations: {max_iter}")
        print(f"Target precision: {precision} eV")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")

    def update_hubbard_u(self, new_u):
        """Update Hubbard parameter in all input files"""
        print(f"Updating U parameter to {new_u:.4f} eV in input files...")
        for filename in self.files:
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                
                updated = False
                with open(filename, 'w') as f:
                    in_hubbard_section = False
                    for line in lines:
                        if 'HUBBARD' in line:
                            in_hubbard_section = True
                            f.write(line)
                            continue
                        
                        if in_hubbard_section:
                            if 'U Mn1-3d' in line:
                                parts = line.split()
                                old_u = float(parts[2])
                                parts[2] = f"{new_u:.4f}"
                                line = ' '.join(parts) + '\n'
                                updated = True
                            elif 'U Mn2-3d' in line:
                                parts = line.split()
                                old_u = float(parts[2])
                                parts[2] = f"{new_u:.4f}"
                                line = ' '.join(parts) + '\n'
                                updated = True
                            elif '/' in line:
                                in_hubbard_section = False
                        
                        f.write(line)
                
                if updated:
                    print(f"  File {filename}: U changed from {old_u} to {new_u:.4f} eV")
            except Exception as e:
                print(f"Error updating file {filename}: {str(e)}")
                return False
        return True

    def run_bands_pp(self, iteration):
        """Run band structure post-processing"""
        input_file = f'{self.compound}_bands.pp.in'
        output_file = f'{self.compound}_bands.pp.out'
    
        print(f"\n{'='*30}")
        print(f"Iteration {iteration}: POST-PROCESSING calculation")
        print(f"Command: bands.x < {input_file} | tee {output_file}")
        print("="*30)
    
        try:
            result = subprocess.run(
                f'bands.x < {input_file} | tee {output_file}',
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
        
            if "JOB DONE" not in result.stdout:
                print("Error: post-processing did not complete successfully!")
                return False # True
        
            print("Band structure post-processing completed successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"Error executing bands.x: {e}")
            return False # True


    def run_calculation(self, calculation_type, iteration):
        """Run Quantum ESPRESSO calculation with detailed logging"""
        input_file = f'{self.compound}.{calculation_type}.in'
        output_file = f'{self.compound}.{calculation_type}.out'
        
        print(f"\n{'='*30}")
        print(f"Iteration {iteration}: {calculation_type.upper()} calculation")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Command: pw.x < {input_file} | tee {output_file}")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print("="*30)
        
        try:
            start_time = datetime.now()
            result = subprocess.run(
                f'pw.x < {input_file} | tee {output_file}',
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if "JOB DONE" not in result.stdout:
                print(f"Error: {calculation_type} calculation did not complete successfully!")
                print("Last 5 lines of output:")
                print('\n'.join(result.stdout.split('\n')[-5:]))
                return False
            
            print(f"Calculation completed successfully in {duration:.1f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing {calculation_type} calculation:")
            print(f"Return code: {e.returncode}")
            print("Error:")
            print(e.stderr)
            return False

    def calculate_band_gap(self):
        """Calculate band gap with error checking"""
        try:
            # Load band structure data
            if not os.path.exists(f'{self.compound}.bands.dat.gnu'):
                raise FileNotFoundError("Band structure data file not found!")
                
            data = np.loadtxt(f'{self.compound}.bands.dat.gnu')
            k = np.unique(data[:, 0])
            bands = np.reshape(data[:, 1], (-1, len(k)))
            
            # Extract Fermi level
            fermi_energy = None
            if os.path.exists(f'{self.compound}.nscf.out'):
                with open(f'{self.compound}.nscf.out', 'r') as f:
                    for line in f:
                        if "the Fermi energy is" in line:
                            fermi_energy = float(line.split()[4])
                            break
            
            if fermi_energy is None:
                print("Failed to determine Fermi level!")
                return None
            
            # Find VBM and CBM
            vbm = np.max(bands[bands < fermi_energy])
            cbm = np.min(bands[bands > fermi_energy])
            band_gap = cbm - vbm
            
            print(f"Band structure analysis results:")
            print(f"  Fermi level: {fermi_energy:.4f} eV")
            print(f"  VBM: {vbm:.4f} eV, CBM: {cbm:.4f} eV")
            print(f"  Band gap: {band_gap:.4f} eV")
            
            return band_gap
            
        except Exception as e:
            
            def find_min_difference(filename, a):
                
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"File {filename} not found")
                
                with open(filename, 'r') as file:
                    content = file.read()
                
                start_marker = "------ SPIN UP ------------"
                end_marker = "Writing all to output data dir"
                
                data_part = content.split(start_marker)[1].split(end_marker)[0]
                
                blocks = data_part.split('k = ')[1:]
                min_difference = float('inf')
                
                for block in blocks:
                    lines = block.split('\n')
                    numbers = []
                    
                    for line in lines[1:]: 
                        cleaned_line = line.strip()
                        if cleaned_line:
                            numbers.extend([float(num) for num in cleaned_line.split() 
                                          if num.replace('.', '').replace('-', '').isdigit()])
                    
                    lower = -float('inf')
                    upper = float('inf')
                    
                    for num in sorted(numbers):
                        if num < a:
                            lower = num
                        elif num > a and upper == float('inf'):
                            upper = num
                            break 
                    
                    if lower != -float('inf') and upper != float('inf'):
                        difference = upper - lower
                        if difference < min_difference:
                            min_difference = difference
                
                return min_difference               
        
            fermi_energy = None
            if os.path.exists(f'{self.compound}.nscf.out'):
                with open(f'{self.compound}.nscf.out', 'r') as f:
                    for line in f:
                        if "the spin up/dw Fermi energies are" in line:
                            fermi_energy = float(line.split()[6])
                            break
            
            if fermi_energy is None:
                print("Failed to determine Fermi level!")
                return None
            
            #print(f"Error calculating band gap: {str(e)}")
            filename = f'{self.compound}.band.out'
            band_gap = find_min_difference(filename, fermi_energy) 
            print(f"Band structure analysis results:")
            print(f"  Fermi level: {fermi_energy:.4f} eV")
            print(f"  Band gap: {band_gap:.4f} eV")            
            return band_gap
   

    def objective_function(self, u, iteration):
        """Objective function with full logging"""
        print(f"\n{'#'*60}")
        print(f"Iteration {iteration}: start (U = {u:.4f} eV)")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#"*60)
        
        # Update U parameter
        if not self.update_hubbard_u(u):
            return -np.inf
        
        # Sequence of calculations
        calculations = ['scf', 'nscf', 'band']
        for calc in calculations:
            if not self.run_calculation(calc, iteration):
                return -np.inf
                
        # Post-processing
        if not self.run_bands_pp(iteration):
            return -np.inf
        
        # Analyze results
        band_gap = self.calculate_band_gap()
        if band_gap is None:
            return -np.inf
        
        # Save history
        self.u_history.append(u)
        self.gap_history.append(band_gap)
        
        # Calculate objective function
        deviation = abs(band_gap - self.target_gap)
        score = -(deviation ** 2)
        
        print(f"\nIteration {iteration} results:")
        print(f"  Band gap: {band_gap:.4f} eV")
        print(f"  Deviation from target: {deviation:.4f} eV")
        print(f"  Score: {score:.4f}")
        print(f"{'#'*60}\n")
        
        return score

    def bayesian_optimization(self):
        """Main optimization function with precision-based stopping"""
        print("\n" + "="*60)
        print("Starting Bayesian optimization")
        print(f"Max iterations: {self.max_iter}")
        print(f"Target precision: {self.precision} eV")
        print("="*60 + "\n")
        
        for i, u in enumerate(self.initial_u):
            initial_score = self.objective_function(u, i)
            if initial_score == -np.inf:
                print(f"\n!!! Error in initial point {u}! Skipping.")
                continue
        
        # Main loop
        for i in range(1, self.max_iter+1):
            print(f"\n{'='*60}")
            print(f"Iteration {i}/{self.max_iter}")
            print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            # Check if we've reached the desired precision
            if len(self.gap_history) > 0 and abs(self.gap_history[-1] - self.target_gap) < self.precision:
                print(f"\nPrecision target reached! (Error < {self.precision} eV)")
                break
                
            # Search space
            u_space = np.linspace(*self.bounds, 1000)
            
            # Update Gaussian process
            X_train = np.array(self.u_history)
            y_train = np.array([-(gap - self.target_gap)**2 for gap in self.gap_history])
            mu, cov = update_gaussian_process(X_train, y_train, u_space)
            
            ##########################################################
            best_y = np.max(y_train)
            std_dev = np.sqrt(np.diag(cov))
            
            # xi = np.clip(0.3 * (1 - (i/30)**2), 0.01, 0.3)  
            xi = 0.01
            
            with np.errstate(divide='ignore', invalid='ignore'): 
                Z = (mu - best_y - xi) / std_dev
                Z[std_dev == 0] = 0 
            
            EI = (mu - best_y - xi) * norm.cdf(Z) + std_dev * norm.pdf(Z)
            EI[std_dev == 0] = 0 
            
            next_idx = np.argmax(EI)
            next_u = u_space[next_idx]
            
            ##########################################################
            """

            # Select next point (UCB)
            kappa = 1.96  # 99% confidence interval
            std_dev = np.sqrt(np.diag(cov))
            ucb = mu + kappa * std_dev
            next_idx = np.argmax(ucb)
            next_u = u_space[next_idx]
            print(f"Expected improvement: {ucb[next_idx]:.4f}")
            """

            print(f"Proposed next point: U = {next_u:.4f} eV")
            print(f"Expected improvement: {EI[next_idx]:.4f}")
            
            # Save iteration plots
            plot_iteration(self, i, u_space)
            
            # Save iteration files
            save_iteration_files(self, i)
            
            # Calculate at new point
            score = self.objective_function(next_u, i)
            
            # Update current U value
            if score != -np.inf:
                self.current_u = next_u
        
        # Final results
        self.show_final_results()
        save_to_npz(self)

    def show_final_results(self):
        """Display final results"""
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        best_idx = np.argmin(np.abs(np.array(self.gap_history) - self.target_gap))
        best_u = self.u_history[best_idx]
        best_gap = self.gap_history[best_idx]
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Total execution time: {total_time:.1f} minutes")
        print(f"Total iterations completed: {len(self.u_history)}")
        print(f"Best U parameter found: {best_u:.4f} eV")
        print(f"Corresponding band gap: {best_gap:.4f} eV")
        print(f"Deviation from target: {abs(best_gap - self.target_gap):.4f} eV")
        print("="*60 + "\n")
        
        plot_final_results(self)