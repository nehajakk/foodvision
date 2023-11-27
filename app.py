import datetime
import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import os
import torch
from model import create_effnetb2_model

#from utils import load_and_prep, preprocess_img, get_classes

with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in f.readlines()]

# Create model
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)


# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
	map_location=torch.device("cpu"),  # load to CPU

    )
)
#def preprocess_img(image):
 # image = tf.image.decode_jpeg(image, channels=3)
  #image = tf.image.convert_image_dtype(image, tf.uint8)
  #return image

#def load_and_prep(image, img_shape=224):
#  image = preprocess_img(image)
#  image = tf.image.resize(image, [img_shape, img_shape])
#  return tf.cast(image, tf.float32)

#@st.cache_resource(suppress_st_warning=True)#
def predicting(image):
    #image = load_and_prep(image)
    img = effnetb2_transforms(image).unsqueeze(0)
    effnetb2.eval()
    with torch.inference_mode():
      # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
      pred_probs = torch.softmax(effnetb2(img), dim=1)

    #predictions = model.predict(tf.expand_dims(image, axis=0))
    pred_class = class_names[tf.argmax(pred_probs, axis=1)[0].numpy()]
    pred_conf = tf.reduce_max(pred_probs[0])
    #top_5 = sorted((predictions.argsort())[0][-5:][::-1])
    #values = predictions[0][top_5] * 100
    #labels = []
    #for x in range(5):
        #labels.append(class_names[top_5[x]])
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {
      class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }
    values = np.array([])

    labels = sorted(pred_labels_and_probs.keys(),reverse=True,key=lambda x : pred_labels_and_probs[x])[:5]    
    for k in labels:
      values = np.append(values, pred_labels_and_probs[k])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

st.set_page_config(page_title="Food Vision",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Food Vision ?")
st.sidebar.write("""
FoodVision is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 

It can identify over 100 different food classes

It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

**Accuracy :** **`80% +`**

**Model :** **`EfficientNetB0`**

**Dataset :** **`Food101`**
""")


#### Main Body ####

st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")
st.write("To know more about this app, visit [**GitHub**](https://github.com/boradj/food-vision)")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])



#model = tf.keras.models.load_model("./07_efficientnetb0_fine_tuned_101_classes_mixed_precision_80_validation.h5")


st.sidebar.markdown("Created by **Jaydip Borad**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://twitter.com/jdborad" target="blank"><img align="center" src="https://bit.ly/3wK17I6" alt="jdborad" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://www.linkedin.com/in/jaydip-borad/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="jaydipborad" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/boradj" target="blank"><img align="center" src="https://bit.ly/githubjd" alt="boradj" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://www.instagram.com/borad.j/" target="blank"><img align="center" src="https://bit.ly/3oZABHZ" alt="borad.j" height="40" width="40" /></a></th>

""", unsafe_allow_html=True)

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    #image = file.read()
    image = Image.open(file)
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))