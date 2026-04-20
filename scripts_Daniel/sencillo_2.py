from ase.build import molecule
from mace.calculators import mace_polar

import numpy as np


atoms = molecule("H2O")
atoms.info["charge"] = 0.0
atoms.info["spin"] = 0.0


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
from matplotlib.widgets import Slider

# 1. Reformar la densidad a 3D
rho_3D = rho.reshape((N_puntos, N_puntos, N_puntos))

# Calculamos el límite GLOBAL para la escala de colores. 
# Esto asegura que el rojo y el azul signifiquen exactamente 
# la misma cantidad de carga en cualquier plano Z.
vlim_global = np.max(np.abs(rho_3D))

# Configuramos la figura dejando un espacio abajo para el slider
fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
plt.subplots_adjust(bottom=0.25) # Espacio inferior

# Índice inicial: empezamos en el centro de masa de la molécula
centro_masa = np.mean(position, axis=0)
indice_z_actual = np.argmin(np.abs(z - centro_masa[2]))

# 2. Plot Inicial
im = ax.imshow(rho_3D[:, :, indice_z_actual].T, origin='lower', 
               extent=[x[0], x[-1], y[0], y[-1]], 
               cmap='bwr', vmin=-vlim_global, vmax=vlim_global, interpolation='bilinear')

# 3. Configurar los átomos y sus transparencias
# Para actualizar los átomos fluidamente, creamos un array RGBA donde variamos el canal Alpha
distancias = np.abs(position[:, 2] - z[indice_z_actual])
alphas = np.exp(-distancias**2 / (2 * sigma**2))

# Color de relleno (negro) y borde (blanco)
rgba_fill = np.zeros((len(position), 4))
rgba_fill[:, 3] = np.clip(alphas, 0, 1)

rgba_edge = np.zeros((len(position), 4))
rgba_edge[:, :3] = 1.0 
rgba_edge[:, 3] = np.clip(alphas, 0, 1)

scatter = ax.scatter(position[:, 0], position[:, 1], 
                     facecolors=rgba_fill, edgecolors=rgba_edge, 
                     s=180, zorder=3)

# Guardamos referencias a los textos para poder cambiarles el Alpha luego
textos_atomos = []
simbolos = atoms.get_chemical_symbols()
for i, pos in enumerate(position):
    txt = ax.text(pos[0]+0.15, pos[1]+0.15, simbolos[i], 
                  fontsize=10, fontweight='bold', zorder=4, 
                  alpha=np.clip(alphas[i], 0, 1))
    textos_atomos.append(txt)

# Detalles de la gráfica
plt.colorbar(im, ax=ax, label=r'Densidad de Carga Proyectada $\rho(\mathbf{r})$')
titulo = ax.set_title(f'Corte de Densidad de Carga\nPlano Z = {z[indice_z_actual]:.3f} Å', fontsize=13)
ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.5)

# ==========================================
# 4. Creación del Slider
# ==========================================
# Ejes para el slider: [posición_x, posición_y, ancho, alto]
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
z_slider = Slider(
    ax=ax_slider,
    label='Eje Z (Å)',
    valmin=z[0],
    valmax=z[-1],
    valinit=z[indice_z_actual]
)

# Función que se ejecuta cada vez que mueves el slider
def update(val):
    # Encontrar el índice Z más cercano al valor seleccionado en el slider
    z_val = z_slider.val
    idx = np.argmin(np.abs(z - z_val))
    
    # Actualizar la matriz de densidad mostrada
    im.set_data(rho_3D[:, :, idx].T)
    
    # Recalcular las transparencias de los átomos
    nuevas_dist = np.abs(position[:, 2] - z[idx])
    nuevos_alphas = np.exp(-nuevas_dist**2 / (2 * sigma**2))
    nuevos_alphas = np.clip(nuevos_alphas, 0, 1)
    
    # Actualizar colores del scatter
    nuevo_rgba_fill = np.zeros((len(position), 4))
    nuevo_rgba_fill[:, 3] = nuevos_alphas
    
    nuevo_rgba_edge = np.zeros((len(position), 4))
    nuevo_rgba_edge[:, :3] = 1.0 
    nuevo_rgba_edge[:, 3] = nuevos_alphas
    
    scatter.set_facecolors(nuevo_rgba_fill)
    scatter.set_edgecolors(nuevo_rgba_edge)
    
    # Actualizar transparencias de los textos
    for i, txt in enumerate(textos_atomos):
        txt.set_alpha(nuevos_alphas[i])
        
    # Actualizar el título dinámicamente
    titulo.set_text(f'Corte de Densidad de Carga\nPlano Z = {z[idx]:.3f} Å')
    
    # Avisar a la figura que debe redibujarse
    fig.canvas.draw_idle()

# Conectar el slider con la función de actualización
z_slider.on_changed(update)

plt.show()