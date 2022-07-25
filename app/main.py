import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()

from PIL import Image
from helper import *
st.title('Alzheimers Classifier')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0        

uploaded_file = st.file_uploader("Upload Image")

print("File uploaded")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        display_image = display_image.resize((500,300))
        st.image(display_image)

        prediction = predictor(os.path.join('uploaded',uploaded_file.name))
        print(prediction)
        
        st.text('**Predictions**')
        fig, ax = plt.subplots()
        ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)
        ax.set(xlabel='Confidence %', ylabel='Classification')
        st.pyplot(fig)
