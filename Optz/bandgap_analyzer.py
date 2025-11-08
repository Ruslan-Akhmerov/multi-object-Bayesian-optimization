import os

class BandgapAnalyzer:
    def __init__(self, compound):
        self.compound = compound
    
    def calculate_band_gap(self, fermi_level):
        """Calculate band gap from band.out file using Fermi level"""
        band_out_file = f'{self.compound}.band.out'
        if not os.path.exists(band_out_file):
            raise FileNotFoundError(f"Band output file {band_out_file} not found")
        
        with open(band_out_file, 'r') as file:
            content = file.read()
        
        start_marker = "------ SPIN UP ------------"
        end_marker = "Writing all to output data dir"
        
        try:
            data_part = content.split(start_marker)[1].split(end_marker)[0]
        except IndexError:
            raise ValueError("Could not parse band structure data from output file")
        
        blocks = data_part.split('k = ')[1:]
        min_difference = float('inf')
        
        for block in blocks:
            lines = block.split('\n')
            numbers = []
            

            for line in lines[1:]:
                cleaned_line = line.strip()
                if cleaned_line:
                    numbers.extend([
                        float(num) for num in cleaned_line.split() 
                        if self._is_float(num)
                    ])
            
            lower = -float('inf')
            upper = float('inf')
            
            for num in sorted(numbers):
                if num < fermi_level:
                    lower = num
                elif num > fermi_level and upper == float('inf'):
                    upper = num
                    break 
            
            if lower != -float('inf') and upper != float('inf'):
                difference = upper - lower
                if difference < min_difference:
                    min_difference = difference
        
        if min_difference == float('inf'):
            raise ValueError("Could not determine band gap from the data")
        
        print("Band structure analysis results:")
        print(f"  Fermi level: {fermi_level:.4f} eV")
        print(f"  Band gap: {min_difference:.4f} eV")
        
        return min_difference
    
    def _is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False