from ase.build import molecule
from mace.calculators import mace_polar

import numpy as np


atoms = molecule("H2O")
atoms.info["charge"] = 0.0
atoms.info["spin"] = 1.0


calc = mace_polar(
    model="polar-1-m",
    device="cpu",           # or "cuda"
    default_dtype="float64" # use float32 for faster MD
)

atoms.calc = calc
position = atoms.get_positions()
print("\nAtomic Positions: ", position)
print("\nShape positions: ", position.shape)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print("\nEnergy:", energy)
print("\nForces:", forces)
print("\nStress:", stress)

# ------------ Total Dipole

dipole = calc.results["dipole"]
print("\nTotal Dipole:", dipole)

# ------------ Density Coefficients

p = calc.results["density_coefficients"]
print("\nDensity Coefficients:", p)
print("\nShape p:", p.shape)

sigma = calc.models[0].atomic_multipoles_smearing_width
print("\nSigma:", sigma)

from prueba_2 import calcular_densidad_rho

L = 6.0 # Tamaño de la caja (12x12x12 Angstroms)
N_puntos = 80 # Puntos por eje (80x80x80 = 512,000 puntos)

x = np.linspace(-L, L, N_puntos)
y = np.linspace(-L, L, N_puntos)
z = np.linspace(-L, L, N_puntos)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

puntos_malla_3d = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

rho = calcular_densidad_rho(puntos_malla_3d, position, p, sigma)

import matplotlib.pyplot as plt

# 1. Reformar la densidad a 3D
rho_3D = rho.reshape((N_puntos, N_puntos, N_puntos))

# 2. Centrado Automático (Universal para cualquier molécula)
# Calculamos el centro de masa simple (promedio de posiciones)
centro_masa = np.mean(position, axis=0)
z_corte_val = centro_masa[2]

# Encontrar el índice de la malla más cercano al centro de masa en Z
indice_z_corte = np.argmin(np.abs(z - z_corte_val))
corte_2d = rho_3D[:, :, indice_z_corte]

# 3. Configuración del Gráfico
fig, ax = plt.subplots(figsize=(9, 7), dpi=100)

# Escala de colores: bwr es ideal para densidad de carga porque 
# resalta zonas de exceso (+) y déficit (-) de carga respecto al promedio.
vlim = np.max(np.abs(corte_2d))

im = ax.imshow(corte_2d.T, origin='lower', 
               extent=[x[0], x[-1], y[0], y[-1]], 
               cmap='bwr', vmin=-vlim, vmax=vlim, interpolation='bilinear')

# 4. Superposición de Átomos Dinámica
# Dibujamos todos los átomos, pero con transparencia según su distancia al plano
distancias = np.abs(position[:, 2] - z[indice_z_corte])
for i, pos in enumerate(position):
    # Átomos muy lejos del plano se ven más tenues
    alpha_val = np.exp(-distancias[i]**2 / (2 * sigma**2)) 
    if alpha_val > 0.1: # Solo mostrar si están relativamente cerca
        ax.scatter(pos[0], pos[1], c='black', edgecolors='white', 
                   s=180, alpha=alpha_val, zorder=3)
        ax.text(pos[0]+0.15, pos[1]+0.15, atoms.get_chemical_symbols()[i], 
                fontsize=10, fontweight='bold', zorder=4)

# 5. Detalles finales
plt.colorbar(im, ax=ax, label=r'Densidad de Carga Proyectada $\rho(\mathbf{r})$')
ax.set_title(f'Corte de Densidad de Carga - Modelo MACE-POLAR\nPlano Z = {z[indice_z_corte]:.3f} Å (Centro de Masa)', fontsize=13)
ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()