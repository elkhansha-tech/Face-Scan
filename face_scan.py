import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
kernel = np.ones((5, 5), np.uint8)
green = (0, 255, 120)
t, speed = 0.0, 0.03
def put(img, text, org, color=(255, 0, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def scan_box(frame, x, y, w, h, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
    ly = y + int(t * h)
    cv2.line(frame, (x, ly), (x + w, ly), green, 2)
    cv2.putText(frame, label, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)


while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

    t += speed
    if t >= 1 or t <= 0:
        speed = -speed

    # --- лицо ---
    for (x, y, w, h) in faces[:1]:
        scan_box(frame, x, y, w, h, "FACE SCANNING...")

    # --- руки: ищем участки цвета кожи ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 70), (20, 170, 255))

    # вырезаем из маски лицо и шею, чтобы они не считались рукой
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x - 30, y - 50), (x + w + 30, y + h + 120), 0, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 8000]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        scan_box(frame, x, y, w, h, "HAND SCANNING...")
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1)
            cv2.drawContours(frame, [c], -1, green, 2)
            cv2.drawContours(frame, [hull], -1, (255, 220, 0), 1)
            cv2.circle(frame, (cx, cy), 6, green, -1)

            # считаем пальцы по впадинам между ними
            fingers = 0
            try:
                idx = cv2.convexHull(c, returnPoints=False)
                defects = cv2.convexityDefects(c, idx)
                if defects is not None:
                    gaps = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        st, en, fa = tuple(c[s][0]), tuple(c[e][0]), tuple(c[f][0])
                        a = math.dist(st, en)
                        b = math.dist(st, fa)
                        cc = math.dist(en, fa)
                        if b * cc == 0:
                            continue
                        cosv = max(-1, min(1, (b * b + cc * cc - a * a) / (2 * b * cc)))
                        if math.degrees(math.acos(cosv)) < 90 and d / 256.0 > 20:
                            gaps += 1
                    fingers = min(gaps + 1, 5) if gaps else 0
            except Exception:
                pass

            lines = [
                "A = %d px^2" % area,
                "P = %.1f px" % perim,
                "C = (%d, %d)" % (cx, cy),
                "r = sqrt(A/pi) = %.1f" % math.sqrt(area / math.pi),
                "A/A_hull = %.2f" % solidity,
                "4*pi*A/P^2 = %.2f" % (4 * math.pi * area / (perim ** 2)),
                "fingers n = %d" % fingers,
                "sin(t) = %.2f" % math.sin(t * math.pi),
            ]
            px = x + w + 10
            if px + 190 > frame.shape[1]:
                px = max(x - 200, 5)
            for i, text in enumerate(lines):
                put(frame, text, (px, y + 20 + i * 20))

    cv2.imshow("Face Scan", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import math