import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 75

# Coba beberapa indeks kamera jika 2 tidak berfungsi
camera_index = 0
for idx in [0, 1, 2, 3]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        # Cek apakah bisa baca frame
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            camera_index = idx
            cap.release()
            break
        else:
            cap.release()

print(f"Menggunakan kamera dengan indeks {camera_index}")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # Lewati jika frame tidak valid

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # Lewati frame yang tidak valid

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('s'):  # Opsional: tambahkan tombol untuk skip frame
            continue

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        print(f"Saved image {counter} for class {j}")

cap.release()
cv2.destroyAllWindows()
