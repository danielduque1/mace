import numpy as np

def calcular_densidad_rho(puntos_malla, r_atomos, coeficientes_p, sigma):
    """
    Calcula la densidad total rho(r) superponiendo las bases ponderadas por p_{i,lm}.
    Eq 32: rho(r) = sum_{i, lm} p_{i,lm} * phi_nlm(r - r_i)
    
    Parámetros:
    puntos_malla   : array (M, 3) - Puntos de la malla 3D.
    r_atomos       : array (n, 3) - Posiciones de los n átomos.
    coeficientes_p : array (n, 4) - Coeficientes [p_00, p_1m1, p_10, p_11] por átomo.
    sigma          : float - Ancho de la gaussiana.
    
    Retorna:
    array (M,) - La densidad evaluada en cada punto de la malla.
    """
    # 1. Distancias relativas (M, n, 3)
    delta_r = puntos_malla[:, np.newaxis, :] - r_atomos[np.newaxis, :, :]
    
    dist_sq = np.sum(delta_r**2, axis=2)
    exponencial = np.exp(-dist_sq / (2 * sigma**2))
    
    # 2. Constantes
    C_10 = 2.0 / (np.pi**0.25 * sigma**1.5)
    Y_00 = np.sqrt(1.0 / (4.0 * np.pi))
    
    C_11 = np.sqrt(8.0/3.0) / (np.pi**0.25 * sigma**2.5)
    factor_Y1 = np.sqrt(3.0 / (4.0 * np.pi))
    
    # 3. Bases por átomo (Forma: M, n)
    phi_00   = C_10 * exponencial * Y_00
    phi_1m1  = C_11 * exponencial * factor_Y1 * delta_r[:, :, 1]  # y
    phi_10   = C_11 * exponencial * factor_Y1 * delta_r[:, :, 2]  # z
    phi_11   = C_11 * exponencial * factor_Y1 * delta_r[:, :, 0]  # x
    
    # 4. Multiplicamos por los coeficientes y sumamos sobre los n átomos (axis=1)
    # Broadcasting: (M, n) * (n,) funciona perfectamente alineando los átomos
    rho = np.sum(
        phi_00  * coeficientes_p[:, 0] +
        phi_1m1 * coeficientes_p[:, 1] +
        phi_10  * coeficientes_p[:, 2] +
        phi_11  * coeficientes_p[:, 3],
        axis=1
    )
    
    return rho

def exportar_cubo(filename, origen, nx, ny, nz, dx, dy, dz, posiciones, z_atomicos, densidad_3d):
    """
    Exporta la malla volumétrica al formato Gaussian .cube.
    Convierte internamente de Angstroms a Radios de Bohr.
    """
    BOHR = 1.8897259886 # 1 Angstrom = 1.8897259886 Bohr
    
    with open(filename, 'w') as f:
        # 1. Encabezado del archivo cube (2 líneas de comentarios)
        f.write("Densidad proyectada de orbitales/features\n")
        f.write("Generado con Python\n")
        
        # 2. Número de átomos y origen de la malla (convertido a Bohr)
        f.write(f"{len(posiciones)} {origen[0]*BOHR:13.6f} {origen[1]*BOHR:13.6f} {origen[2]*BOHR:13.6f}\n")
        
        # 3. Vectores de la malla: (N_puntos, paso_x, paso_y, paso_z) en Bohr
        f.write(f"{nx} {dx*BOHR:13.6f} 0.000000 0.000000\n")
        f.write(f"{ny} 0.000000 {dy*BOHR:13.6f} 0.000000\n")
        f.write(f"{nz} 0.000000 0.000000 {dz*BOHR:13.6f}\n")
        
        # 4. Información atómica: (Num_Atómico, Carga_Efectiva, X, Y, Z) en Bohr
        for pos, z_num in zip(posiciones, z_atomicos):
            f.write(f"{z_num} {float(z_num):13.6f} {pos[0]*BOHR:13.6f} {pos[1]*BOHR:13.6f} {pos[2]*BOHR:13.6f}\n")
        
        # 5. Escribir los datos volumétricos
        # El formato cube requiere iterar X exterior, Y medio, Z interior.
        # Imprime máximo 6 valores por línea.
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{densidad_3d[i, j, k]:13.5e} ")
                    if (k + 1) % 6 == 0:
                        f.write("\n")
                if nz % 6 != 0:
                    f.write("\n")

# ==========================================
# Ejecución principal
# ==========================================
if __name__ == "__main__":
    sigma_1 = 1.0
    
    # Sistema de prueba: una molécula de agua (H2O) simulada
    posiciones_atomos = np.array([
        [ 0.000,  0.000,  0.117], # Oxígeno
        [ 0.000,  0.757, -0.469], # Hidrógeno 1
        [ 0.000, -0.757, -0.469]  # Hidrógeno 2
    ])
    numeros_atomicos = [8, 1, 1] # Z para O, H, H
    
    # Coeficientes p_{i, lm} (n=3 átomos, 4 canales)
    # Forma: [p_00, p_1-1(y), p_10(z), p_11(x)]
    # Estos son valores inventados para que veas la forma en VESTA. 
    # En tu proyecto, estos saldrán de tu modelo.
    coeficientes = np.array([
        [1.5,  0.0, -0.5, 0.0], # Átomo 0: Fuerte s, algo de p_z negativo
        [0.8,  0.3,  0.2, 0.0], # Átomo 1: Mezcla s y p
        [0.8, -0.3,  0.2, 0.0]  # Átomo 2: Mezcla s y p (simétrico en y)
    ])
    
    # Configuración de la malla 3D (Caja de 6x6x6 Angstroms)
    L = 3.0 
    N_puntos = 60 # Puntos por eje (60x60x60 = 216,000 puntos)
    
    # Usamos indexing='ij' para que la matriz (Nx, Ny, Nz) se alinee 
    # correctamente con los ejes X, Y, Z al momento de exportar
    x = np.linspace(-L, L, N_puntos)
    y = np.linspace(-L, L, N_puntos)
    z = np.linspace(-L, L, N_puntos)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    puntos_malla_3d = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Calcular pasos espaciales (dx, dy, dz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    origen = [x[0], y[0], z[0]]
    
    print("Calculando densidad 3D sobre la malla...")
    rho_1D = calcular_densidad_rho(puntos_malla_3d, posiciones_atomos, coeficientes, sigma_1)
    
    # Reformar al volumen 3D
    rho_3D = rho_1D.reshape((N_puntos, N_puntos, N_puntos))
    
    print("Exportando archivo .cube...")
    exportar_cubo("densidad_proyectada.cube", origen, N_puntos, N_puntos, N_puntos, 
                  dx, dy, dz, posiciones_atomos, numeros_atomicos, rho_3D)
    
    print("¡Listo! Ya puedes abrir 'densidad_proyectada.cube' en VESTA o ChimeraX.")