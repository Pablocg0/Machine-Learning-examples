import numpy as np
"""Clase que contendra funciones basicas para
el uso de los datos que se usaran en otras clases"""
class MachineLearningUtils:
    def __init__(self):
        self.datos= []

    #Funcion que normaliza los datos
    #con la formula norm(x)= x - promedio / desviacionEstandar
    #recibe una arreglo del tipo ndarray
    #regresa una tupla (B,C)
    #B  es un arreglo con los valores normalizados
    #C es  un arreglo de tamano 2xn donde n es el numero de columnas del arreglo original
    #en la posicion 1xn se guarda el promedio de la columna y en la posicion 2xn se guarda
    #el valor de la desviacionEstandar de la columna
    def normData(self, A):
        row= 0 #variable para el control de la fila del arreglo se encuentras
        column =0 #variable para el control de la columna se encuentras
        size = A.shape[1] # Numero de filas del arreglo original
        sizeColumn =A.shape[0] #Numero de columnas de arreglo original
        C = np.zeros([2,sizeColumn]) #Arreglo del tamano 2xn
        B = np.zeros(A.shape) #Arreglo del mismo tamano que el original
        for list in A:
            for x in list:
                mod = row%size
                prom = np.mean(A[column:column+1]) # promedio del columna
                stdesv= np.std(A[column:column+1]) # desviacionEstandar de la columna
                norm = (x - prom)/stdesv
                C[0,column] = prom #guardamos el promdio
                C[1,column] = stdesv #guardamos la desviacionEstandar
                B[column,mod]= prom #guardamos el valor normalizados
                row = row +1
            column = column + 1
        return (B,C)
