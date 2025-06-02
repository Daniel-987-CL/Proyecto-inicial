import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo entrenado
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

# Iniciar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detección de rostro con MediaPipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Validar que sea una imagen RGB válida
        if frame_rgb.dtype != np.uint8 or frame_rgb.shape[2] != 3:
            print("Imagen inválida para MediaPipe")
            continue

        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * width)
                ymin = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Verificar que el bounding box no se salga del frame
                if xmin < 0 or ymin < 0 or xmin + w > width or ymin + h > height:
                    continue

                face_image = frame[ymin:ymin + h, xmin:xmin + w]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)

                result = face_mask.predict(face_image)

                if result[1] < 150:
                    label = LABELS[result[0]]
                    color = (0, 255, 0) if label == "Con_mascarilla" else (0, 0, 255)

                    cv2.putText(frame, label, (xmin, ymin - 15), 2, 1, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
