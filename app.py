import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("가위바위보 분류기")

# 모델과 라벨 불러오기
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r", encoding="utf-8") as f:
    labels = f.readlines()

# 카메라
photo = st.camera_input("가위, 바위, 보를 보여주세요")

if photo is not None:
    image = Image.open(photo).convert("RGB")

    # 224 x 224 크기로 변환
    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    # Teachable Machine 모델에 맞게 정규화
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image

    # 예측
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = labels[index].strip()

    # 앞의 숫자 제거
    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]

    confidence = prediction[0][index]

    st.subheader("분류 결과")
    st.write(f"### {class_name}")
    st.write(f"정확도: {confidence * 100:.1f}%")