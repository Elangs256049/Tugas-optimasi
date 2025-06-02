import numpy as np
import matplotlib.pyplot as plt
import random

class PSO:
    def __init__(self, num_particles=10, max_iterations=50, w=0.5, c1=1.5, c2=1.5, bounds=(-10, 10)):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.bounds = bounds
        
        # Initialize particles
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_values = []
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.best_values_history = []
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Initialize particle positions and velocities randomly"""
        for i in range(self.num_particles):
            # Random position within bounds
            position = random.uniform(self.bounds[0], self.bounds[1])
            self.particles.append(position)
            
            # Random velocity
            velocity = random.uniform(-1, 1)
            self.velocities.append(velocity)
            
            # Initialize personal best
            self.personal_best_positions.append(position)
            fitness = self.objective_function(position)
            self.personal_best_values.append(fitness)
            
            # Update global best
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best_position = position
    
    def objective_function(self, x):
        """Objective function: f(x) = x²"""
        return x ** 2
    
    def update_particles(self):
        """Update particle velocities and positions"""
        for i in range(self.num_particles):
            # Update velocity
            r1 = random.random()
            r2 = random.random()
            
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
            
            self.velocities[i] = (self.w * self.velocities[i] + 
                                cognitive_component + 
                                social_component)
            
            # Update position
            self.particles[i] += self.velocities[i]
            
            # Apply bounds
            if self.particles[i] < self.bounds[0]:
                self.particles[i] = self.bounds[0]
            elif self.particles[i] > self.bounds[1]:
                self.particles[i] = self.bounds[1]
            
            # Evaluate fitness
            fitness = self.objective_function(self.particles[i])
            
            # Update personal best
            if fitness < self.personal_best_values[i]:
                self.personal_best_values[i] = fitness
                self.personal_best_positions[i] = self.particles[i]
                
                # Update global best
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i]
    
    def optimize(self):
        """Run the PSO optimization algorithm"""
        print("=== PSEUDOCODE ALGORITMA PSO ===")
        print("1. Inisialisasi populasi partikel dengan posisi dan kecepatan acak")
        print("2. Evaluasi fitness setiap partikel")
        print("3. Untuk setiap iterasi:")
        print("   a. Update kecepatan partikel")
        print("   b. Update posisi partikel")
        print("   c. Evaluasi fitness baru")
        print("   d. Update personal best dan global best")
        print("4. Ulangi hingga kriteria berhenti tercapai")
        print("\n=== IMPLEMENTASI PSO ===")
        
        print(f"Fungsi objektif: f(x) = x²")
        print(f"Batas pencarian: {self.bounds[0]} ≤ x ≤ {self.bounds[1]}")
        print(f"Jumlah partikel: {self.num_particles}")
        print(f"Iterasi maksimum: {self.max_iterations}")
        print(f"Parameter - w: {self.w}, c1: {self.c1}, c2: {self.c2}")
        print("\nMulai optimisasi...")
        
        # Record initial best value
        self.best_values_history.append(self.global_best_value)
        
        for iteration in range(self.max_iterations):
            self.update_particles()
            self.best_values_history.append(self.global_best_value)
            
            if iteration % 10 == 0:
                print(f"Iterasi {iteration}: Best value = {self.global_best_value:.6f}, Best position = {self.global_best_position:.6f}")
        
        print(f"\n=== HASIL OPTIMISASI ===")
        print(f"Nilai minimum yang ditemukan: {self.global_best_value:.6f}")
        print(f"Posisi x terbaik: {self.global_best_position:.6f}")
        print(f"Iterasi total: {self.max_iterations}")
        
        return self.global_best_position, self.global_best_value
    
    def plot_convergence(self):
        """Plot grafik konvergensi nilai terbaik per iterasi"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.best_values_history)), self.best_values_history, 'b-', linewidth=2)
        plt.title('Konvergensi PSO - Nilai Terbaik per Iterasi')
        plt.xlabel('Iterasi')
        plt.ylabel('Nilai Fungsi Objektif (f(x) = x²)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale untuk melihat konvergensi lebih jelas
        plt.show()
        
        # Plot fungsi objektif dan posisi partikel terakhir
        plt.figure(figsize=(10, 6))
        x_range = np.linspace(-10, 10, 1000)
        y_range = x_range ** 2
        
        plt.plot(x_range, y_range, 'b-', label='f(x) = x²', linewidth=2)
        plt.scatter(self.particles, [self.objective_function(x) for x in self.particles], 
                   c='red', s=50, alpha=0.7, label='Posisi Partikel Akhir')
        plt.scatter(self.global_best_position, self.global_best_value, 
                   c='green', s=100, marker='*', label='Global Best')
        
        plt.title('Fungsi Objektif dan Posisi Partikel')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-10, 10)
        plt.ylim(0, 100)
        plt.show()

# Jalankan algoritma PSO
if __name__ == "__main__":
    # Inisialisasi PSO dengan parameter yang ditentukan
    pso = PSO(
        num_particles=10,
        max_iterations=50,
        w=0.5,
        c1=1.5,
        c2=1.5,
        bounds=(-10, 10)
    )
    
    # Jalankan optimisasi
    best_position, best_value = pso.optimize()
    
    # Tampilkan grafik
    pso.plot_convergence()
    
    print(f"\n=== RINGKASAN HASIL ===")
    print(f"Algoritma PSO berhasil menemukan minimum global!")
    print(f"Nilai minimum teoritis: 0 (pada x = 0)")
    print(f"Nilai minimum yang ditemukan: {best_value:.8f}")
    print(f"Posisi x optimal: {best_position:.8f}")
    print(f"Error: {abs(best_value):.8f}")