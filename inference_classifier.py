import cv2
import mediapipe as mp
import numpy as np
import pickle

# Muat model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {i: str(i) for i in range(10)}

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

expected_features = 42  # 21 landmarks × 2 koordinat (x, y)

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_numbers = []  # Untuk menyimpan angka yang terdeteksi dari masing-masing tangan
    bounding_boxes = []    # Untuk menyimpan posisi kotak agar bisa tampilkan info tangan

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Gambar landmark
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2))

            # Ekstraksi landmark
            data_aux = []
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Sesuaikan jumlah fitur
            if len(data_aux) < expected_features:
                data_aux += [0.0] * (expected_features - len(data_aux))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            # Prediksi
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])
                predicted_label = labels_dict.get(predicted_index, "Unknown")

                # Cek apakah ini angka 0-9
                if predicted_label.isdigit():
                    detected_numbers.append(int(predicted_label))

                # Gambar kotak dan teks
                x1 = max(int(min_x * W) - 20, 0)
                y1 = max(int(min_y * H) - 20, 0)
                x2 = min(int(max_x * W) + 20, W)
                y2 = min(int(max_y * H) + 20, H)

                bounding_boxes.append((x1, y1, x2, y2, predicted_label))

            except Exception as e:
                print("Prediction error:", e)

        # Gambar kotak dan label setelah semua tangan diproses
        for box in bounding_boxes:
            x1, y1, x2, y2, label = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Tampilkan hasil penjumlahan jika ada dua angka
    if len(detected_numbers) >= 2:
        sum_result = sum(detected_numbers)
        result_text = f"Hasil Penjumlahan: {sum_result}"
        cv2.putText(frame, result_text, (20, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow('Sign Language Detection - Penjumlahan Dua Tangan', frame)

    if cv2.waitKey(1) == 27:  # Tekan ESC untuk keluar
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()