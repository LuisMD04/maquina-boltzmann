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
    """
    delta_e = np.dot(pesos[indice], estado) + sesgos[indice]
    exponente = np.clip(-delta_e / temperatura, -500, 500)
    probabilidad = 1.0 / (1.0 + np.exp(exponente))
    return probabilidad

def actualizar_unidad(indice, estado, pesos, sesgos, temperatura, generador):
    """
    Actualiza el estado de una neurona específica de manera estocástica.
    """
    probabilidad = probabilidad_activacion(indice, estado, pesos, sesgos, temperatura)
    numero_aleatorio = generador.uniform(0, 1)
    
    if numero_aleatorio < probabilidad:
        estado[indice] = 1
    else:
        estado[indice] = 0
        
    return estado

def simular(estado_inicial, pesos, sesgos, temperatura, iteraciones, semilla):
    """
    Ejecuta múltiples iteraciones de actualización, registra la dinámica y exporta a CSV.
    """
    comprobar_matriz(pesos)
    generador = np.random.RandomState(semilla)
    estado_actual = np.copy(estado_inicial)
    num_unidades = len(estado_actual)
    
    historial_estados = []
    historial_energias = []
    
    for _ in range(iteraciones):
        indice = generador.randint(0, num_unidades)
        estado_actual = actualizar_unidad(indice, estado_actual, pesos, sesgos, temperatura, generador)
        
        estado_str = "".join(str(int(x)) for x in estado_actual)
        e_actual = energia(estado_actual, pesos, sesgos)
        
        historial_estados.append(estado_str)
        historial_energias.append(e_actual)
        
    df = pd.DataFrame({
        'Iteracion': range(1, iteraciones + 1),
        'Estado': historial_estados,
        'Energia': historial_energias
    })
    
    nombre_archivo = f'resultados_T_{temperatura}.csv'
    df.to_csv(nombre_archivo, index=False)
    print(f"Simulación finalizada. Resultados guardados en: {nombre_archivo}")
    
    return df

if __name__ == '__main__':
    # --- Parámetros iniciales ---
    W = np.array([
        [0.0, 0.8, -0.4],
        [0.8, 0.0, 0.6],
        [-0.4, 0.6, 0.0]
    ])
    b = np.array([0.2, -0.1, 0.3])
    estado_inicial = np.array([0, 0, 0])
    temperatura = 1.0
    iteraciones = 10000
    semilla = 42 
    
    # --- Ejecución de la simulación ---
    df_resultados = simular(estado_inicial, W, b, temperatura, iteraciones, semilla)
    
    # --- 8.3 Comparación teórica y experimental ---
    probabilidades_teoricas = {
        "000": 0.0698, "001": 0.0942, "010": 0.0632, "011": 0.1554,
        "100": 0.0853, "101": 0.0772, "110": 0.1717, "111": 0.2831
    }
    
    burn_in = 1000
    df_util = df_resultados.iloc[burn_in:].copy()
    
    total_muestras = len(df_util)
    frecuencias_exp = df_util['Estado'].value_counts() / total_muestras
    
    print("\n--- Tabla de Comparación Teórica vs Experimental ---")
    print(f"{'Estado':<8} | {'P. Teórica':<12} | {'Frec. Exp.':<12} | {'Error Absoluto':<14}")
    print("-" * 55)
    
    for estado, p_teorica in probabilidades_teoricas.items():
        p_exp = frecuencias_exp.get(estado, 0.0)
        error_abs = abs(p_teorica - p_exp)
        print(f"{estado:<8} | {p_teorica:<12.4f} | {p_exp:<12.4f} | {error_abs:<14.4f}")
