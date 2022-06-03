import streamlit as st
import cv2
import numpy as np
from PIL import Image
from canny_edge_module import make_edge

def pil2cv(img):
    img_new = np.array(img, dtype=np.uint8)
    if img_new.ndim == 2 : #gray scale
        pass
    elif img_new.shape[2] == 3 : #color
        img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
    elif img_new.shape[2] == 4 : #transparent
        img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGRA)
    return img_new

st.title('Multiple Files Upload Demo')

st.sidebar.title('Image files uploader')

uploaded_files = st.sidebar.file_uploader('Please upload images',
                                  type=['png','jpg','jpeg'],
                                  accept_multiple_files=True)

#Class
#edge parameters
kernel1=7
thres1_val=30
thres2_val=9
kernel2=2
iter_dialate=3
iter_erode=2

edge = make_edge.edge(kernel1=kernel1, thres1_val=thres1_val, thres2_val=thres2_val,
                      kernel2=kernel2, iter_dialate=iter_dialate, iter_erode=iter_erode)

# img_ret1, img_ret2 = edge.preprocess(img_array=img_src, rot_state=False)


for upload_file in uploaded_files:


    image = Image.open(upload_file)
    st.image(image)

    img_ret1, _ = edge.preprocess(img_array=pil2cv(image), rot_state=False, trans_state=False)
    st.image(img_ret1)





