import cv2

# Inicializar a captura de vídeo da webcam (0 é a câmera padrão)
cap = cv2.VideoCapture(0)

# Inicializar o primeiro frame (frame de referência)
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    # Capturar um novo frame
    ret, frame2 = cap.read()
    if not ret:
        break
    
    # Converta o frame atual para escala de cinza
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calcule a diferença absoluta entre o frame atual e o frame de referência
    frame_diff = cv2.absdiff(prev_gray, gray)
    
    # Aplicar uma limiarização para destacar as áreas em movimento
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos nas áreas em movimento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar retângulos ao redor das áreas em movimento
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Exibir o frame com as áreas em movimento detectadas
    cv2.imshow("Detector de Movimento", frame2)
    
    # Atualizar o frame de referência
    prev_gray = gray.copy()
    
    # Pressione a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
