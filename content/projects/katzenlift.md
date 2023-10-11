---
title: Katzenlift
tags:
  - Tensorflow
  - Object Detection
  - Image Classification
  - Python
cover:
  image: "cover/mauzi.jpg"
  alt: "Bild nicht gefunden"
  relative: true
---
# Automatischer Katzenlift
Genau genommen ist es kein Lift sondern eine Treppe, welche heruntergefahren werden soll, sobald eine zur Wohnung zugehörige Katze von der Kamera entdeckt wird.
In diesem Blog geht es jedoch nur um die Programmierung der Erkennungsmodelle.
## Vorgehen
1. [Erstellung eines Proof of Concept](#1-erstellung-eines-proof-of-concept)
2. [Daten Sammeln](#2-trainings-und-validierungs-daten-sammeln)
3. [Erstes Trainieren und anwenden]()
4. [Object Detection hinzufügen]()
5. []

## 1 Erstellung eines Proof of Concept
[Github Repository für POC](https://github.com/toky03/teachable_image_recognition)
Wir wollen nicht das Rad neu erfinden. Deshalb müssen wir nicht ein komplett neues Tensorflow Model von Grund auf neu bauen.
Aus diesem Grund entscheiden wir uns erstmal für ein Transfer-learning und lehnen uns auch beim POC an ein [offizieles Tensorflow Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning) speziell für Transfer Learning mit Bildern.
Somit habe ich zuerst 1:1 das Tutorial durchgespielt.
### Transfer Learning
Das Transfer Learning bedeutet, dass ein bestehenden Model in diesem Fall [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) welches direkt via Keras heruntergeladen wird.
Da im [Tutorial von Tensorflow](https://www.tensorflow.org/tutorials/images/transfer_learning) bereits sehr gut darauf eingegangen wird, wie genau das Transferlearning durchgeführt verzichte ich hierbei auf eine detailierte beschreibung und gebe lediglich eine kurze Zusammenfassung.
#### Kurz Zusammenfassung Tutorial
Im Tutorial wird ein vorhandenes Model weitertrainiert, damit es Bilder in die Kategorie Hund oder Katze einteilen kann.
Hierfür werden [beispiel Bilder von Katzen und Hunden](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip) heruntergeladen und danach in eine spezielle Ordnerstruktur gebracht ein Ordner `train` für das Training und ein Ordner `validation` für die Validierung nach jeder Epoche. Jeder dieser Ordner hat die selbe struktur, nämlich weitere Unterordner für die jeweilige Kategorie **Hund** und **Katze**.
Mit dem sehr hilfreichen keras Packet `keras.utils.image_dataset_from_directory` kann dann ein ganzer Ordner als dataset geladen werden
Der nächste wichtige Schritt ist data augmentation. Hier werden die einzelnen Bilder verfielfältigt indem diese Horizontal gespiegelt und zufällig rotiert werden.
```python
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])
```
```python
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
```
Danach wird das basis Model heruntergeladen und die weiteren Schichten darauf aufgebaut:
```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

```python
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
```

## 2 Trainings und Validierungs Daten sammeln
Nun da das Tutorial durchgespielt wurde, können wir uns daran setzten das ganze auf unsere eigenen Ansprüche zu modifizieren.
...WIP
