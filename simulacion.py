import numpy as np
import pandas as pd

def comprobar_matriz(pesos):
    """Comprueba que la matriz de pesos sea simétrica y su diagonal sea cero."""
    es_simetrica = np.allclose(pesos, pesos.T)
    diagonal_cero = np.all(np.diag(pesos) == 0)
    
    if not es_simetrica:
        raise ValueError("Error: La matriz de pesos no es simétrica.")
    if not diagonal_cero:
        raise ValueError("Error: La diagonal de la matriz de pesos no es cero.")
    return True

def energia(estado, pesos, sesgos):
    """
    Calcula la energía del estado actual de la red.
    E(s) = -0.5 * sum_i sum_j W_ij s_i s_j - sum_i b_i s_i
    """
    termino_pesos = -0.5 * np.dot(estado.T, np.dot(pesos, estado))
    termino_sesgos = -np.dot(sesgos.T, estado)
    return termino_pesos + termino_sesgos

def probabilidad_activacion(indice, estado, pesos, sesgos, temperatura):
    """
    Calcula la probabilidad de que una neurona específica se active (s_i = 1).
    P(s_i=1) = 1 / (1 + e^(-Delta E_i / T))
    Delta E_i se calcula como la suma de las conexiones de la neurona i más su sesgo.
    """
    # Calculamos Delta E_i (influencia sobre la neurona i)
    delta_e = np.dot(pesos[indice], estado) + sesgos[indice]
    
    # Prevenimos el desbordamiento (overflow) limitando el exponente
    exponente = np.clip(-delta_e / temperatura, -500, 500)
    
    probabilidad = 1.0 / (1.0 + np.exp(exponente))
    return probabilidad

def actualizar_unidad(indice, estado, pesos, sesgos, temperatura, generador):
    """
    Actualiza el estado de una neurona específica de manera estocástica.
    """
    probabilidad = probabilidad_activacion(indice, estado, pesos, sesgos, temperatura)
    
    # Generamos un número pseudoaleatorio entre 0 y 1 utilizando el generador
    numero_aleatorio = generador.uniform(0, 1)
    
    # Si el número es menor que la probabilidad, la neurona se activa (1), si no se apaga (0)
    if numero_aleatorio < probabilidad:
        estado[indice] = 1
    else:
        estado[indice] = 0
        
    return estado

def simular(estado_inicial, pesos, sesgos, temperatura, iteraciones, semilla):
    """
    Ejecuta múltiples iteraciones de actualización, registra la dinámica y exporta a CSV.
    """
    # Comprobar restricciones de la matriz
    comprobar_matriz(pesos)
    
    # Documentar e inicializar la semilla aleatoria para reproducibilidad
    generador = np.random.RandomState(semilla)
    
    # Crear una copia independiente del estado inicial
    estado_actual = np.copy(estado_inicial)
    num_unidades = len(estado_actual)
    
    # Listas para almacenar las frecuencias y los resultados
    historial_estados = []
    historial_energias = []
    
    for _ in range(iteraciones):
        # Muestreo de Gibbs: elegimos una neurona al azar para actualizar
        indice = generador.randint(0, num_unidades)
        estado_actual = actualizar_unidad(indice, estado_actual, pesos, sesgos, temperatura, generador)
        
        # Formatear el estado como cadena (ej. "101") y calcular la energía de la iteración
        estado_str = "".join(str(int(x)) for x in estado_actual)
        e_actual = energia(estado_actual, pesos, sesgos)
        
        # Registrar resultados
        historial_estados.append(estado_str)
        historial_energias.append(e_actual)
        
    # Exportar las frecuencias observadas y resultados a un archivo CSV
    df = pd.DataFrame({
        'Iteracion': range(1, iteraciones + 1),
        'Estado': historial_estados,
        'Energia': historial_energias
    })
    
    nombre_archivo = f'resultados_T_{temperatura}.csv'
    df.to_csv(nombre_archivo, index=False)
    print(f"Simulación finalizada. Resultados guardados en: {nombre_archivo}")
    
    return df

# --- Bloque principal (Prueba utilizando los datos de la Sección 4.3) ---
if __name__ == '__main__':
    # Datos de prueba según el ejercicio matemático obligatorio
    W = np.array([
        [0.0, 0.8, -0.4],
        [0.8, 0.0, 0.6],
        [-0.4, 0.6, 0.0]
    ])
    
    b = np.array([0.2, -0.1, 0.3])
    
    estado_inicial = np.array([0, 0, 0])
    temperatura = 1.0
    iteraciones = 10000
    semilla = 42 # Semilla aleatoria documentada
    
    # Llamada a la función de simulación
    df_resultados = simular(estado_inicial, W, b, temperatura, iteraciones, semilla)
