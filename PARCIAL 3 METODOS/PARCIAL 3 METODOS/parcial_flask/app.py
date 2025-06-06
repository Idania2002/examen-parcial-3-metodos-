from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta absoluta a la carpeta actual (donde está app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta absoluta a la carpeta 'static'
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Crea la carpeta 'static' si no existe
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

from scipy.interpolate import lagrange, KroghInterpolator
from numpy.polynomial.polynomial import Polynomial

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    ecuacion = ""
    valor_x = None
    # Esta variable indicará si hay resultados para mostrar (y por lo tanto, la opción de ver gráfico)
    hay_resultados = False 

    if request.method == "POST":
        metodo = request.form["metodo"]
        x = list(map(float, request.form["x"].split(",")))
        y = list(map(float, request.form["y"].split(",")))
        valor_x = float(request.form["valor_x"])
        
        # Siempre generamos el gráfico aquí
        resultado, ecuacion = aplicar_metodo(metodo, x, y, valor_x)
        hay_resultados = True # Hay resultados, así que podemos preguntar sobre el gráfico

    # Pasamos 'hay_resultados' a la plantilla
    return render_template("index.html", resultado=resultado, ecuacion=ecuacion, valor_x=valor_x, hay_resultados=hay_resultados)

# NOTA: Quité el parámetro 'mostrar_grafico' de aquí, el gráfico siempre se generará
def aplicar_metodo(metodo, x, y, x_valor): 
    xp = np.array(x)
    yp = np.array(y)

    plt.figure() # Siempre creamos la figura
    plt.scatter(xp, yp, color="blue", label="Datos")

    if metodo == "regresion_lineal":
        coef = np.polyfit(xp, yp, 1)
        p = np.poly1d(coef)
        y_pred = p(x_valor)
        ecuacion = f"y = {coef[0]:.3f}x + {coef[1]:.3f}"
        plt.plot(xp, p(xp), color="red", label="Regresión Lineal")

    elif metodo == "regresion_cuadratica":
        coef = np.polyfit(xp, yp, 2)
        p = np.poly1d(coef)
        y_pred = p(x_valor)
        ecuacion = f"y = {coef[0]:.3f}x² + {coef[1]:.3f}x + {coef[2]:.3f}"
        plt.plot(xp, p(xp), color="green", label="Regresión Cuadrática")

    elif metodo == "interpolacion_lineal":
        y_pred = np.interp(x_valor, xp, yp)
        ecuacion = "Interpolación lineal aplicada"
        x_interp = np.linspace(min(xp), max(xp), 100)
        y_interp = np.interp(x_interp, xp, yp)
        plt.plot(x_interp, y_interp, color="red", label="Interpolación Lineal")

    elif metodo == "lagrange":
        poly = lagrange(xp, yp)
        y_pred = poly(x_valor)
        ecuacion = "Lagrange: " + str(poly)
        x_interp = np.linspace(min(xp), max(xp), 100)
        y_interp = poly(x_interp)
        plt.plot(x_interp, y_interp, color="red", label="Interpolación Lagrange")

    elif metodo == "extrapolacion_lineal":
        p = np.poly1d(np.polyfit(xp, yp, 1))
        y_pred = p(x_valor)
        ecuacion = f"y = {p[0]:.3f}x + {p[1]:.3f}"
        min_x_plot = min(xp) - abs(max(xp) - min(xp)) * 0.1
        max_x_plot = max(xp) + abs(max(xp) - min(xp)) * 0.1
        x_plot = np.linspace(min_x_plot, max_x_plot, 100)
        plt.plot(x_plot, p(x_plot), color="orange", label="Extrapolación Lineal")

    elif metodo == "newton":
        interpolador = KroghInterpolator(xp, yp)
        y_pred = interpolador(x_valor)
        ecuacion = "Extrapolación por Newton (Krogh): polinomio aplicado"
        min_x_plot = min(xp) - abs(max(xp) - min(xp)) * 0.1
        max_x_plot = max(xp) + abs(max(xp) - min(xp)) * 0.1
        x_interp = np.linspace(min_x_plot, max_x_plot, 100)
        y_interp = interpolador(x_interp)
        plt.plot(x_interp, y_interp, color="red", label="Extrapolación Newton")

    # Siempre guardamos el gráfico
    plt.title("Gráfico")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(STATIC_DIR, "graph.png"))
    plt.close()

    return y_pred, ecuacion

if __name__ == "__main__":
    app.run(debug=True)