import cv2
from PIL import Image, ImageEnhance, UnidentifiedImageError
import joblib
import numpy as np
from skimage.feature import hog


# Inicializar la cámara
lda = joblib.load("modelo_entrenado.pkl")
kmeans_lda = joblib.load("modelo_entrenado2.pkl")
cap = cv2.VideoCapture(0)

def extract_features(image):
    image = np.array(image)
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def preprocess_image(image):
    try:
        # Convertir la matriz de imagen en una imagen PIL
        image_pil = Image.fromarray(image)

        # Redimensionar imagen
        image_resized = image_pil.resize((128, 128))

        # Convertir a escala de grises
        image_grayscale = image_resized.convert('L')

        # Ajustar contraste
        enhancer = ImageEnhance.Contrast(image_grayscale)
        image_contrast = enhancer.enhance(2)  # Aumentar contraste

        return image_contrast
    except Exception as e:
        print(f"Error en la preprocesamiento de la imagen: {e}")
        return None

while True:
    # Capturar frame de la cámara
    ret, frame = cap.read()
    
    # Preprocesar el frame
    processed_frame = preprocess_image(frame)
    
    if processed_frame is not None:
        # Extraer características del frame preprocesado
        frame_features = extract_features(processed_frame)
        
        # Transformar las características del frame utilizando LDA
        transformed_features = lda.transform(frame_features.reshape(1, -1))
        
        # Clasificar el frame en un cluster utilizando KMeans entrenado con LDA
        cluster = kmeans_lda.predict(transformed_features)
        
        # Mostrar el cluster asignado en el frame
        cv2.putText(frame, f'Cluster: {cluster}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el frame
    cv2.imshow('Frame', frame)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()