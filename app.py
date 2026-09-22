import os
from flask import Flask, request, jsonify
import keras
import numpy as np
from PIL import Image

app = Flask(__name__)

MODELO_PATH = "modelo_final.keras"

print("Cargando modelo Dermelia...")

modelo = keras.models.load_model(MODELO_PATH)

print("MODELO CARGADO CORRECTAMENTE")
print("Entrada:", modelo.input_shape)
print("Salida:", modelo.output_shape)


@app.after_request
def habilitar_cors(response):
    # Permite que el frontend (abierto como archivo local, o publicado en
    # otro dominio) pueda llamar a esta API sin que el navegador lo bloquee.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def inicio():
    return "API Dermelia funcionando correctamente"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predecir():
    if request.method == "OPTIONS":
        # El navegador manda este pedido "de aviso" antes del POST real,
        # para confirmar que el CORS está permitido. Solo hay que responder
        # OK vacío, sin procesar nada.
        return "", 204

    try:
        # Verificar que llegó una imagen
        if "imagen" not in request.files:
            return jsonify({
                "error": "No se recibió ninguna imagen."
            }), 400

        archivo = request.files["imagen"]

        # Abrir y preparar imagen
        imagen = Image.open(archivo).convert("RGB")
        imagen = imagen.resize((224, 224))

        arr = np.array(imagen, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        # Predicción
        probabilidad = float(
            modelo.predict(arr, verbose=0)[0][0]
        )

        # Clasificación
        if probabilidad >= 0.65:
            resultado = "Posiblemente maligno"
        elif probabilidad > 0.35:
            resultado = "Resultado poco claro"
        else:
            resultado = "Posiblemente benigno"

        return jsonify({
            "probabilidad_malignidad": probabilidad,
            "porcentaje_malignidad": round(probabilidad * 100, 1),
            "porcentaje_benignidad": round((1 - probabilidad) * 100, 1),
            "resultado": resultado
        })

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "error": "No se pudo analizar la imagen."
        }), 500

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
