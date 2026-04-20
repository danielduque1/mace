# import matplotlib.pyplot as plt
# import numpy as np

# def evaluar_base_lm_completa(r, r_i, sigma):
#     """
#     Evalúa simultáneamente los 4 canales (l=0, m=0) y (l=1, m=-1, 0, 1)
#     usando armónicos sólidos reales.
    
#     Parámetros:
#     r     : array - Posición(es) de evaluación.
#     r_i   : array - Posiciones de los átomos de referencia.
#     sigma : float - Ancho de la gaussiana (sigma_1).
    
#     Retorna:
#     array (N, 4) - Matriz donde cada columna es un canal lm:
#                    Col 0: (0, 0)
#                    Col 1: (1,-1) -> y
#                    Col 2: (1, 0) -> z
#                    Col 3: (1, 1) -> x
#     """
#     # Vector de distancias relativas
#     delta_r = r - r_i
    
#     # Manejar caso de un solo punto vs múltiples puntos para robustez
#     if delta_r.ndim == 1:
#         delta_r = delta_r.reshape(1, 3)
        
#     # Distancia al cuadrado (r^2)
#     dist_sq = np.sum(delta_r**2, axis=1)
    
#     # Parte exponencial compartida por los 4 canales
#     exponencial = np.exp(-dist_sq / (2 * sigma**2))
    
#     # --- Constantes Precalculadas ---
#     # Para l=0
#     C_10 = 2.0 / (np.pi**0.25 * sigma**1.5)
#     Y_00 = np.sqrt(1.0 / (4.0 * np.pi))
    
#     # Para l=1
#     C_11 = np.sqrt(8.0/3.0) / (np.pi**0.25 * sigma**2.5)
#     factor_Y1 = np.sqrt(3.0 / (4.0 * np.pi))
    
#     # --- Evaluación de los canales ---
#     # Canal 0: (0, 0)
#     canal_00 = C_10 * exponencial * Y_00
    
#     # Canales 1: (1, -1), (1, 0), (1, 1) proporcionales a y, z, x respectivamente
#     factor_comun_l1 = C_11 * exponencial * factor_Y1
    
#     canal_1_m1 = factor_comun_l1 * delta_r[:, 1]  # y
#     canal_1_0  = factor_comun_l1 * delta_r[:, 2]  # z
#     canal_1_1  = factor_comun_l1 * delta_r[:, 0]  # x
    
#     # Apilamos en una matriz (N, 4)
#     return np.column_stack((canal_00, canal_1_m1, canal_1_0, canal_1_1))

# # 1. Configuración de la malla
# sigma = 1.5
# r_atomo = np.array([[0.0, 0.0, 0.0]]) 

# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)

# puntos_malla = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# # 2. Evaluación vectorizada (¡Solo 1 llamada para los 4 canales!)
# base_lm = evaluar_base_lm_completa(puntos_malla, r_atomo, sigma)

# # base_lm tiene forma (10000, 4). Extraemos los canales:
# val_00   = base_lm[:, 0]  # l=0, m=0
# val_1_m1 = base_lm[:, 1]  # l=1, m=-1 (p_y)
# val_1_0  = base_lm[:, 2]  # l=1, m=0  (p_z)
# val_1_1  = base_lm[:, 3]  # l=1, m=1  (p_x)

# # Reconstruir formas (100x100)
# Grid_00   = val_00.reshape(X.shape)
# Grid_1_m1 = val_1_m1.reshape(X.shape)
# Grid_1_1  = val_1_1.reshape(X.shape)

# # 3. Visualización (Omitimos p_z porque en el plano XY evaluado en Z=0, p_z es cero)
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# titulos = [r'Canal 0: $(0,0)$', r'Canal 3: $(1,1) \rightarrow x$', r'Canal 1: $(1,-1) \rightarrow y$']
# mallas = [Grid_00, Grid_1_1, Grid_1_m1]

# vmax = max(np.max(np.abs(Grid_1_1)), np.max(np.abs(Grid_00)))

# for ax, malla, titulo in zip(axes, mallas, titulos):
#     c = ax.contourf(X, Y, malla, levels=50, cmap='bwr', vmin=-vmax, vmax=vmax)
#     ax.set_title(titulo)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_aspect('equal')
#     fig.colorbar(c, ax=ax)

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def evaluar_superposicion_base(puntos_malla, r_atomos, sigma):
    """
    Evalúa la superposición de la base para n átomos sobre M puntos de una malla.
    
    Parámetros:
    puntos_malla : array (M, 3) - Puntos del espacio donde se evalúa.
    r_atomos     : array (n, 3) - Posiciones de los n átomos.
    sigma        : float - Ancho de la gaussiana.
    
    Retorna:
    array (M, 4) - Matriz con los 4 canales evaluados sumando la contribución de los n átomos.
    """
    # 1. Broadcasting para obtener todas las distancias cruzadas
    # puntos_malla adquiere forma (M, 1, 3)
    # r_atomos adquiere forma (1, n, 3)
    # delta_r tendrá forma (M, n, 3): El vector r - r_i para cada punto y cada átomo
    delta_r = puntos_malla[:, np.newaxis, :] - r_atomos[np.newaxis, :, :]
    
    # Distancia al cuadrado y exponencial (Forma: M, n)
    dist_sq = np.sum(delta_r**2, axis=2)
    exponencial = np.exp(-dist_sq / (2 * sigma**2))
    
    # 2. Constantes de normalización
    C_10 = 2.0 / (np.pi**0.25 * sigma**1.5)
    Y_00 = np.sqrt(1.0 / (4.0 * np.pi))
    
    C_11 = np.sqrt(8.0/3.0) / (np.pi**0.25 * sigma**2.5)
    factor_Y1 = np.sqrt(3.0 / (4.0 * np.pi))
    
    # 3. Evaluación de los canales por átomo (Forma: M, n)
    canal_00_por_atomo   = C_10 * exponencial * Y_00
    canal_1_m1_por_atomo = C_11 * exponencial * factor_Y1 * delta_r[:, :, 1]  # y
    canal_1_0_por_atomo  = C_11 * exponencial * factor_Y1 * delta_r[:, :, 2]  # z
    canal_1_1_por_atomo  = C_11 * exponencial * factor_Y1 * delta_r[:, :, 0]  # x
    
    # 4. Superposición: Sumamos a lo largo del eje de los átomos (axis=1)
    # Esto reduce la dimensión de (M, n) a (M,) para cada canal
    total_00   = np.sum(canal_00_por_atomo, axis=1)
    total_1_m1 = np.sum(canal_1_m1_por_atomo, axis=1)
    total_1_0  = np.sum(canal_1_0_por_atomo, axis=1)
    total_1_1  = np.sum(canal_1_1_por_atomo, axis=1)
    
    return np.column_stack((total_00, total_1_m1, total_1_0, total_1_1))

# ==========================================
# Configuración del sistema
# ==========================================
sigma_1 = 1.5

# Array de posiciones de prueba (n, 3)
# Puedes reemplazar esto por tu propia estructura de datos
posiciones_atomos = np.array([
    [-2.0,  0.0, 0.0],
    [ 2.0,  0.0, 0.0],
    [ 0.0,  2.5, 0.0]
])

dist_z = 1.0

# Crear malla 2D en el plano XY (Z=0)
limite = 6.0
x = np.linspace(-limite, limite, 150)
y = np.linspace(-limite, limite, 150)
X, Y = np.meshgrid(x, y)
Z = np.ones_like(X) * dist_z

puntos_malla = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# ==========================================
# Ejecución y reconstrucción
# ==========================================
# Evaluar los 4 canales para todos los átomos
base_lm = evaluar_superposicion_base(puntos_malla, posiciones_atomos, sigma_1)

Grid_00   = base_lm[:, 0].reshape(X.shape) # s
Grid_1_m1 = base_lm[:, 1].reshape(X.shape) # p_y
Grid_1_0  = base_lm[:, 2].reshape(X.shape) # p_z
Grid_1_1  = base_lm[:, 3].reshape(X.shape) # p_x

# ==========================================
# Visualización
# ==========================================
# Haremos un grid de 2x2 para mostrar los 4 canales
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

titulos = [r'Canal 0: $(0,0)$ (Simetría s)', 
           r'Canal 3: $(1,1) \rightarrow x$', 
           r'Canal 1: $(1,-1) \rightarrow y$',
           r'Canal 2: $(1,0) \rightarrow z$']
mallas = [Grid_00, Grid_1_1, Grid_1_m1, Grid_1_0]

# Unificar la escala de colores basada en el valor absoluto máximo global
vmax = max([np.max(np.abs(m)) for m in mallas])
# Prevenir vmax = 0 si todos los valores son nulos (ej. p_z en Z=0)
if vmax == 0: vmax = 1.0 

for i, (ax, malla, titulo) in enumerate(zip(axes, mallas, titulos)):
    c = ax.contourf(X, Y, malla, levels=60, cmap='bwr', vmin=-vmax, vmax=vmax)
    
    # Graficar las posiciones de los átomos encima de la malla
    ax.scatter(posiciones_atomos[:, 0], posiciones_atomos[:, 1], 
               color='black', marker='o', s=50, edgecolor='white', label='Átomos')
    
    ax.set_title(titulo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()