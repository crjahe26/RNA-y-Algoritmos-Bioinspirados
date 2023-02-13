# %%
import numpy as np
import matplotlib.pyplot as plt
import random as random
from scipy.ndimage import convolve
import time

# %%
random.seed(1983)
ncols = 5
nrows = 5
n = ncols*nrows
x = random.choices(population = [0,1], k = n)
arreglo = np.array(x).reshape((nrows,ncols))
print(arreglo)

# %%
kernel = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])
conteo_vecinos = convolve(arreglo, kernel, mode='constant')
print(conteo_vecinos)

# %%
(conteo_vecinos == 2)
# %%
(conteo_vecinos == 2)|(conteo_vecinos == 3)
# %%
# Pasamos una matriz de booleanos a matriz de 0s y 1s
(conteo_vecinos == 2)|(conteo_vecinos == 3) * 1

# %%
# El complemento del arreglo lo encontramos con (-arreglo + 1)
(conteo_vecinos == 2)|(conteo_vecinos == 3) * 1 + (-arreglo + 1)
# %%
nuevo_arreglo = arreglo * ((conteo_vecinos == 2)| (conteo_vecinos == 3) * 1) + (-arreglo + 1) * (conteo_vecinos == 3) * 1
print(nuevo_arreglo)
















# %%
import numpy as np
import matplotlib.pyplot as plt
import random as random
from scipy.ndimage import convolve
import time

# %%
def inicializar_poblacion(ncols, nrows, semilla,prob_vida = 0.5):
    random.seed(semilla)
    n = ncols*nrows
    celulas = random.choices(population = [0,1], weights=([1-prob_vida, prob_vida]), k = n)
    poblacion = np.array(celulas).reshape((nrows,ncols))
    return(poblacion)


# %%
def actualizar_poblacion(A):
    """
    Actualiza la población en A y crea A_siguiente
    de acuerdo con las reglas del juego de la vida de
    Conway
    A: es un arreglo 2D de ceros y unos
    """
    contidad_vecinos = convolve(A, weights = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]]), mode='constant')
    A_siguiente = A * ((contidad_vecinos==2)|(contidad_vecinos==3)*1) + (-A+1) * (contidad_vecinos==3) * 1
    return(A_siguiente)

# %%
def juego_de_la_vida(nrows,ncols,niter, semilla = 1983):
    # Iniciar población:
    poblacion = inicializar_poblacion(ncols = ncols,nrows = nrows, semilla = semilla)
    for i in range(niter):
        plt.matshow(poblacion)
        plt.title("Iteración " + str(i))
        plt.show()
        # agregar un delay
        time.sleep(0.2)
        # plt.close()
        poblacion = actualizar_poblacion(poblacion)


# %%
#juego_de_la_vida(nrows=200,ncols=200,niter=300)
juego_de_la_vida(nrows=10,ncols=10,niter=300)
# %%
