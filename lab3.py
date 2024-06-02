import cv2
import matplotlib.pyplot as plt

cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

image = cv2.imread('../Lab-3/lab3/img/img.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cats = cat_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

for (x, y, w, h) in cats:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, 'Cat', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
