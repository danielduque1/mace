"""
charge_density.py
Módulo para calcular, guardar y visualizar densidades de carga proyectadas
utilizando modelos MACE-POLAR y ASE.
"""

import argparse
import numpy as np
from ase.io import read

# ==========================================
# 1. Funciones Matemáticas Core
# ==========================================
def _calcular_rho_vectorizada(puntos_malla, posiciones, coeficientes_p, sigma):
    """Motor matemático interno optimizado para evaluar la base."""
    delta_r = puntos_malla[:, np.newaxis, :] - posiciones[np.newaxis, :, :]
    
    dist_sq = np.sum(delta_r**2, axis=2)
    exponencial = np.exp(-dist_sq / (2 * sigma**2))
    
    C_10 = 2.0 / (np.pi**0.25 * sigma**1.5)
    Y_00 = np.sqrt(1.0 / (4.0 * np.pi))
    C_11 = np.sqrt(8.0/3.0) / (np.pi**0.25 * sigma**2.5)
    factor_Y1 = np.sqrt(3.0 / (4.0 * np.pi))
    
    phi_00  = C_10 * exponencial * Y_00
    phi_1m1 = C_11 * exponencial * factor_Y1 * delta_r[:, :, 1]  # y
    phi_10  = C_11 * exponencial * factor_Y1 * delta_r[:, :, 2]  # z
    phi_11  = C_11 * exponencial * factor_Y1 * delta_r[:, :, 0]  # x
    
    rho = np.sum(
        phi_00  * coeficientes_p[:, 0] +
        phi_1m1 * coeficientes_p[:, 1] +
        phi_10  * coeficientes_p[:, 2] +
        phi_11  * coeficientes_p[:, 3],
        axis=1
    )
    return rho

# ==========================================
# 2. API Principal
# ==========================================
def get_charge_density(atoms, padding=3.0, N_puntos=80):
    """
    Calcula la densidad de carga 3D a partir de un objeto ASE Atoms.
    La malla se ajusta automáticamente al tamaño de la molécula.
    
    Parámetros:
    atoms    : Objeto ASE Atoms con el calculador MACE-POLAR adjunto.
    padding  : Margen (en Å) a añadir alrededor de los átomos más extremos.
    N_puntos : Resolución (número de puntos por eje).
    
    Retorna:
    diccionario con rho_3D, origen, pasos (dx,dy,dz), y vectores espaciales.
    """
    if atoms.calc is None:
        raise RuntimeError("El objeto atoms no tiene un calculador adjunto.")
        
    if "density_coefficients" not in atoms.calc.results:
        atoms.get_potential_energy()
        
    p = atoms.calc.results["density_coefficients"]
    sigma = atoms.calc.models[0].atomic_multipoles_smearing_width
    posiciones = atoms.get_positions()
    
    # ----------------------------------------------------
    # LÓGICA DE LA CAJA (CELDA vs VACÍO)
    # ----------------------------------------------------
    if atoms.cell.volume > 1e-6:
        print("Celda detectada. Mapeando la malla a los vectores de la celda...")
        cell = atoms.cell.array
        
        # Malla en coordenadas fraccionales (0 a 1)
        fx = np.linspace(0, 1, N_puntos)
        fy = np.linspace(0, 1, N_puntos)
        fz = np.linspace(0, 1, N_puntos)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        
        # Transformación a cartesianas: r = fx*v1 + fy*v2 + fz*v3
        puntos_malla = (FX.ravel()[:, None] * cell[0] + 
                        FY.ravel()[:, None] * cell[1] + 
                        FZ.ravel()[:, None] * cell[2])
        
        origen = [0.0, 0.0, 0.0]
        
        # Los pasos de la malla ahora son vectores 3D
        paso_x = cell[0] / (N_puntos - 1)
        paso_y = cell[1] / (N_puntos - 1)
        paso_z = cell[2] / (N_puntos - 1)
        
        # Para el ploteo 2D interactivo, aproximamos los ejes con las magnitudes
        ejes = (fx * np.linalg.norm(cell[0]), 
                fy * np.linalg.norm(cell[1]), 
                fz * np.linalg.norm(cell[2]))
        
    else:
        print("Sin celda definida. Calculando Bounding Box automático...")
        min_pos = np.min(posiciones, axis=0) - padding
        max_pos = np.max(posiciones, axis=0) + padding
        
        x = np.linspace(min_pos[0], max_pos[0], N_puntos)
        y = np.linspace(min_pos[1], max_pos[1], N_puntos)
        z = np.linspace(min_pos[2], max_pos[2], N_puntos)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        puntos_malla = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        
        origen = [x[0], y[0], z[0]]
        
        # En una caja ortogonal, los vectores de paso son paralelos a los ejes cartesianos
        paso_x = np.array([x[1] - x[0], 0.0, 0.0])
        paso_y = np.array([0.0, y[1] - y[0], 0.0])
        paso_z = np.array([0.0, 0.0, z[1] - z[0]])
        
        ejes = (x, y, z)

    # ----------------------------------------------------
    # CÁLCULO DE LA DENSIDAD
    # ----------------------------------------------------
    rho_1D = _calcular_rho_vectorizada(puntos_malla, posiciones, p, sigma)
    rho_3D = rho_1D.reshape((N_puntos, N_puntos, N_puntos))
    
    return {
        "rho_3D": rho_3D,
        "origen": origen,
        "pasos": (paso_x, paso_y, paso_z), # Ahora siempre son tuplas de vectores (3,)
        "ejes": ejes,
        "posiciones": posiciones,
        "simbolos": atoms.get_chemical_symbols(),
        "numeros_atomicos": atoms.get_atomic_numbers(),
        "sigma": sigma
    }

def save_charge_density(filename, atoms, datos_densidad):
    """
    Guarda la matriz de densidad en formato .cube.
    
    Parámetros:
    filename       : Nombre del archivo de salida (ej. "densidad.cube").
    atoms          : Objeto ASE Atoms.
    datos_densidad : Diccionario retornado por get_charge_density().
    """
    
    BOHR = 1.8897259886
    rho_3D = datos_densidad["rho_3D"]
    origen = datos_densidad["origen"]
    vec_x, vec_y, vec_z = datos_densidad["pasos"] # Extraemos los vectores de paso
    posiciones = datos_densidad["posiciones"]
    z_atomicos = datos_densidad["numeros_atomicos"]
    
    nx, ny, nz = rho_3D.shape
    
    with open(filename, 'w') as f:
        f.write("Densidad de Carga MACE-POLAR\n")
        f.write("Generado con charge_density.py\n")
        # Origen de la malla
        f.write(f"{len(posiciones)} {origen[0]*BOHR:13.6f} {origen[1]*BOHR:13.6f} {origen[2]*BOHR:13.6f}\n")
        
        # Vectores que definen los pasos de la malla tridimensional
        f.write(f"{nx} {vec_x[0]*BOHR:13.6f} {vec_x[1]*BOHR:13.6f} {vec_x[2]*BOHR:13.6f}\n")
        f.write(f"{ny} {vec_y[0]*BOHR:13.6f} {vec_y[1]*BOHR:13.6f} {vec_y[2]*BOHR:13.6f}\n")
        f.write(f"{nz} {vec_z[0]*BOHR:13.6f} {vec_z[1]*BOHR:13.6f} {vec_z[2]*BOHR:13.6f}\n")
        
        # Posiciones atómicas
        for pos, z_num in zip(posiciones, z_atomicos):
            f.write(f"{z_num} {float(z_num):13.6f} {pos[0]*BOHR:13.6f} {pos[1]*BOHR:13.6f} {pos[2]*BOHR:13.6f}\n")
        
        # Datos volumétricos
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{rho_3D[i, j, k]:13.5e} ")
                    if (k + 1) % 6 == 0:
                        f.write("\n")
                if nz % 6 != 0:
                    f.write("\n")

# ==========================================
# 3. Utilidad Interactiva (Opcional)
# ==========================================
def plot_charge_density_slider(datos_densidad):
    """
    Abre una ventana de Matplotlib con un slider interactivo en el eje Z.
    Importa matplotlib localmente para no requerirlo en entornos sin interfaz gráfica.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        print("Matplotlib no está instalado. No se puede visualizar el plot interactivo.")
        return

    rho_3D = datos_densidad["rho_3D"]
    x, y, z = datos_densidad["ejes"]
    posiciones = datos_densidad["posiciones"]
    simbolos = datos_densidad["simbolos"]
    sigma = datos_densidad["sigma"]
    
    vlim_global = np.max(np.abs(rho_3D))
    fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
    plt.subplots_adjust(bottom=0.25)
    
    centro_masa = np.mean(posiciones, axis=0)
    idx_z = np.argmin(np.abs(z - centro_masa[2]))
    
    im = ax.imshow(rho_3D[:, :, idx_z].T, origin='lower', 
                   extent=[x[0], x[-1], y[0], y[-1]], 
                   cmap='bwr_r', vmin=-vlim_global, vmax=vlim_global, interpolation='bilinear')
    
    distancias = np.abs(posiciones[:, 2] - z[idx_z])
    alphas = np.clip(np.exp(-distancias**2 / (2 * sigma**2)), 0, 1)
    
    rgba_fill = np.zeros((len(posiciones), 4))
    rgba_fill[:, 3] = alphas
    rgba_edge = np.zeros((len(posiciones), 4))
    rgba_edge[:, :3] = 1.0 
    rgba_edge[:, 3] = alphas
    
    scatter = ax.scatter(posiciones[:, 0], posiciones[:, 1], 
                         facecolors=rgba_fill, edgecolors=rgba_edge, s=180, zorder=3)
    
    textos = [ax.text(pos[0]+0.15, pos[1]+0.15, sim, fontsize=10, fontweight='bold', 
                      zorder=4, alpha=a) for pos, sim, a in zip(posiciones, simbolos, alphas)]
    
    plt.colorbar(im, ax=ax, label=r'Densidad de Carga Proyectada $\rho(\mathbf{r})$')
    titulo = ax.set_title(f'Corte de Densidad de Carga\nPlano Z = {z[idx_z]:.3f} Å', fontsize=13)
    ax.set_aspect('equal')
    
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    z_slider = Slider(ax_slider, 'Eje Z (Å)', z[0], z[-1], valinit=z[idx_z])
    
    def update(val):
        idx = np.argmin(np.abs(z - z_slider.val))
        im.set_data(rho_3D[:, :, idx].T)
        
        n_alphas = np.clip(np.exp(-np.abs(posiciones[:, 2] - z[idx])**2 / (2 * sigma**2)), 0, 1)
        
        n_fill, n_edge = np.zeros((len(posiciones), 4)), np.zeros((len(posiciones), 4))
        n_fill[:, 3], n_edge[:, 3] = n_alphas, n_alphas
        n_edge[:, :3] = 1.0
        
        scatter.set_facecolors(n_fill)
        scatter.set_edgecolors(n_edge)
        
        for txt, a in zip(textos, n_alphas): txt.set_alpha(a)
        titulo.set_text(f'Corte de Densidad de Carga\nPlano Z = {z[idx]:.3f} Å')
        fig.canvas.draw_idle()

    z_slider.on_changed(update)
    plt.show()

# ==========================================
# 4. Interfaz de Línea de Comandos (CLI)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula y guarda densidades de carga de modelos MACE-POLAR.")
    parser.add_argument("input_file", type=str, help="Ruta al archivo de entrada (.xyz, .traj, etc.)")
    parser.add_argument("--model", type=str, default="polar-1-m", choices=["polar-1-m", "polar-1-l"], 
                        help="Modelo MACE a utilizar (default: polar-1-m)")
    parser.add_argument("--output", "-o", type=str, default="densidad.cube", help="Nombre del archivo de salida .cube")
    parser.add_argument("--padding", "-p", type=float, default=3.0, help="Margen en Ångstroms alrededor de la molécula (default: 3.0)")
    parser.add_argument("--resolution", "-n", type=int, default=80, help="Número de puntos por eje en la malla 3D (default: 80)")
    parser.add_argument("--plot", action="store_true", help="Abre la ventana interactiva para explorar la densidad antes de terminar")
    
    args = parser.parse_args()
    
    print(f"Cargando geometría desde: {args.input_file}")
    atoms = read(args.input_file)
    
    print(f"Inicializando calculador MACE-POLAR ({args.model})...")
    from mace.calculators import mace_polar # Importación tardía para acelerar el help del CLI
    
    calc = mace_polar(model=args.model, device="cpu", default_dtype="float64")
    atoms.calc = calc
    
    print(f"Calculando densidad de carga en malla {args.resolution}x{args.resolution}x{args.resolution}...")
    datos = get_charge_density(atoms, padding=args.padding, N_puntos=args.resolution)
    
    print(f"Guardando archivo {args.output}...")
    save_charge_density(args.output, atoms, datos)
    print("¡Proceso completado exitosamente!")
    
    if args.plot:
        print("Lanzando visualizador interactivo...")
        plot_charge_density_slider(datos)