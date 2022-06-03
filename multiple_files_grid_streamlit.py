import streamlit as st
import cv2
import numpy as np
from PIL import Image
from canny_edge_module import make_edge
import pandas as pd

st.set_page_config(layout="wide")

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


@st.cache
def pil2cv(img):
    img_new = np.array(img, dtype=np.uint8)
    if img_new.ndim == 2:  # gray scale
        pass
    elif img_new.shape[2] == 3:  # color
        img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
    elif img_new.shape[2] == 4:  # transparent
        img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGRA)
    return img_new


st.title('Multiple Files Upload Demo')
st.subheader('Canny Edge Detection with Masking process')

st.sidebar.title('Image files uploader')

uploaded_files = st.sidebar.file_uploader('Please upload images',
                                          type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)

# edge parameters
kernel1 = st.sidebar.slider('Blur kernel size', min_value=0, max_value=10, value=7)
thres1_val = st.sidebar.slider('Canny Threshold 1', min_value=1, max_value=300, value=45)
thres2_val = st.sidebar.slider('Canny Threshold 2', min_value=1, max_value=300, value=9)
kernel2 = st.sidebar.slider('dialate, erode k_mat_size', min_value=0, max_value=10, value=3)
iter_dialate = st.sidebar.slider('iter_d', min_value=0, max_value=20, value=3)
iter_erode = st.sidebar.slider('iter_e', min_value=0, max_value=20, value=2)

rot_state = st.sidebar.selectbox('Would you like to rotate the body?',
                                 (True, False))
st.sidebar.write('You selected:', rot_state)
trans_state = st.sidebar.selectbox('Would you like to translate the body?',
                                   (True, False))
st.sidebar.write('You selected:', trans_state)

# File Download

canny_param_dict = {'blur kernel_size ': kernel1,
                    'Canny Threshold1': thres1_val,
                    'Canny Threshold2': thres2_val,
                    'dialate, erode k_mat_size': kernel2,
                    'iter_d': iter_dialate,
                    'iter_e': iter_erode,

                    }

# dict to dataframe
df = pd.DataFrame(list(canny_param_dict.items()), columns=['param', 'value'])
csv = convert_df(df)
st.dataframe(df)

st.sidebar.download_button(
    label='Download data as CSV',
    data=csv,
    mime='text/csv')

width_fig = st.sidebar.slider('figure width', min_value=50, max_value=1000, value=200)

# Class
edge = make_edge.edge(kernel1=kernel1, thres1_val=thres1_val, thres2_val=thres2_val,
                      kernel2=kernel2, iter_dialate=iter_dialate, iter_erode=iter_erode)

# img_ret1, img_ret2 = edge.preprocess(img_array=img_src, rot_state=False)



st.subheader('Original Images')
idx = 0
for upload_file in uploaded_files:
    cols = st.columns(4)

    # print(upload_file)
    # UploadedFile(id=1, name='000.png', type='image/png', size=1159058)
    # # id
    # print(upload_file.id)
    # #file name
    # print(upload_file.name)

    # image = Image.open(upload_file)
    # st.image(image)
    #
    # img_ret1, _ = edge.preprocess(img_array=pil2cv(image), rot_state=False, trans_state=False)
    # st.image(img_ret1)

    if idx < len(uploaded_files):
        cols[0].image(uploaded_files[idx], width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    if idx < len(uploaded_files):
        cols[1].image(uploaded_files[idx], width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    if idx < len(uploaded_files):
        cols[2].image(uploaded_files[idx], width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    if idx < len(uploaded_files):
        cols[3].image(uploaded_files[idx], width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    else:
        break


st.subheader('Preprocessed Images[0]')
idx = 0
for upload_file in uploaded_files:
    cols = st.columns(4)

    # image = Image.open(upload_file)
    # st.image(image)
    #
    # img_ret1, _ = edge.preprocess(img_array=pil2cv(image), rot_state=False, trans_state=False)
    # st.image(img_ret1)

    if idx < len(uploaded_files):
        cols[0].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[0],
                      width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    if idx < len(uploaded_files):
        cols[1].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[0],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    if idx < len(uploaded_files):
        cols[2].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[0],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    if idx < len(uploaded_files):
        cols[3].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[0],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    else:
        break



st.subheader('Preprocessed Images[1]')
idx = 0
for upload_file in uploaded_files:
    cols = st.columns(4)

    # image = Image.open(upload_file)
    # st.image(image)
    #
    # img_ret1, _ = edge.preprocess(img_array=pil2cv(image), rot_state=False, trans_state=False)
    # st.image(img_ret1)

    if idx < len(uploaded_files):
        cols[0].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[1],
                      width=width_fig, caption=uploaded_files[idx].name)
        idx += 1
    if idx < len(uploaded_files):
        cols[1].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[1],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    if idx < len(uploaded_files):
        cols[2].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[1],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    if idx < len(uploaded_files):
        cols[3].image(edge.preprocess(img_array=pil2cv(Image.open(uploaded_files[idx])),
                                      rot_state=rot_state, trans_state=trans_state)[1],
                      width=width_fig, caption=uploaded_files[idx].name)

        idx += 1
    else:
        break
