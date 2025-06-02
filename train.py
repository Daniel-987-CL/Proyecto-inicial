import cv2
import os
import numpy as np

# Ruta al dataset (modifica si es necesario)
dataPath = r"C:\Users\vp20d\OneDrive\Escritorio\PROYECTO MASCARILLA\Dataset_faces"

labels = []
facesData = []
label = 0

# Obtener lista de carpetas (personas/clases), ordenadas
dir_list = sorted(os.listdir(dataPath))

# Recorrer cada carpeta (una clase por carpeta)
for name_dir in dir_list:
    dir_path = os.path.join(dataPath, name_dir)

    # Verificar si es una carpeta
    if not os.path.isdir(dir_path):
        print(f"Omitiendo: {dir_path} (no es carpeta)")
        continue

    image_count = 0
    for file_name in os.listdir(dir_path):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(dir_path, file_name)
        image = cv2.imread(image_path, 0)

        if image is None:
            print(f"Advertencia: No se pudo leer {image_path}")
            continue

        print(f"Imagen leída: {image_path}")
        facesData.append(image)
        labels.append(label)
        image_count += 1

    print(f"[{name_dir}] -> {image_count} imágenes procesadas.")
    label += 1

print("Total imágenes:", len(facesData))
print("Total etiquetas:", len(set(labels)))

# Verificar si hay suficientes datos para entrenar
if len(facesData) > 1 and len(set(labels)) > 1:
    try:
        face_mask = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Error: 'cv2.face' no está disponible. Asegúrate de instalar 'opencv-contrib-python'.")
        print("Usa: pip install opencv-contrib-python")
        exit()

    print("Entrenando el modelo...")
    face_mask.train(facesData, np.array(labels))
    face_mask.write("face_mask_model.xml")
    print("Modelo almacenado exitosamente.")
else:
    print("No hay suficientes datos para entrenar el modelo (mínimo 2 clases con imágenes).")
