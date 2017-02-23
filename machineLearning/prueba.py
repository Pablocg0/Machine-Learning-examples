import numpy as np
from MachineLearningUtils import MachineLearningUtils
"""Clase para probar la funcion normData  de la clase MachineLearningUtils"""

m = np.ndarray(shape=(10,10), dtype=int, order='F')# se crea el arreglo
mch = MachineLearningUtils() # se crea un objeto de la clase MachineLearningUtils
print mch.normData(m) #Se imprime el resultado de la funcion normData
