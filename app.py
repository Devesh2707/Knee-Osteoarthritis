from matplotlib.pyplot import show
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from saliency import make_saliency_map
from bokeh.plotting import figure

from utils import download_weights

import io

from config import CONFIG

cfg = CONFIG()

labels = [1, 2, 3, 4, 5]

def make_barh(labels, confidence):

    graph = figure(
        title = 'Confidence for each KL-Grade',
        x_axis_label = 'Confidence',
        y_axis_label  = 'KL-Grade',
        plot_width=400, plot_height=400
    )

    y = labels
    right = confidence

    graph.hbar(
        y = y,
        right = right,
        height = 0.4,
        color = 'red',
    )

    return graph

def main():

    st.title('Knee Osterarthritis KL-Grade Estimator.')
    st.sidebar.header('A CNN Classifier to predict KL-Grade of Knee Osteoarthritis')
    st.sidebar.write('\n')
    st.sidebar.write('\n')

    st.sidebar.write("""

##### **Which model to use ?**:

    """)

    model = st.sidebar.radio(
        label = '',
        options=['densenet121', 'densenet161', 'resnext50_32x4d'],
        index=0,
    )

    download_weights(model_name=model)

    st.write('\n')

    file = st.sidebar.file_uploader("Upload Knee X-Ray here", type=['jpg','png','jpeg'])
    if file is not None:
        img_stream = io.BytesIO(file.read())
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        scores = make_saliency_map(img, model)
        scores = scores.detach().cpu().numpy()[0]

        st.write('\n')

        st.sidebar.write("""

    ##### **Make Heatmap:**:

        """)

        hm = st.sidebar.radio(
                label = '',
                options=['Yes', 'No'],
                index=0,
            )

        st.write('\n')
        st.write('\n')

        col1,col2,col3 = st.beta_columns(3)

        with col2:
            st.image(
                image=img,
                caption='Input X-Ray',
                use_column_width='always',
            )

        st.write('\n')
        st.write('\n')

        if hm == 'Yes':
            im = np.asarray(Image.open(f'{cfg.saliency_map}map_hm.jpg').convert('RGB'))
        else:
            im = np.asarray(Image.open(f'{cfg.saliency_map}map.jpg').convert('RGB'))

        c1, c2 = st.beta_columns(2)

        with c1:
            st.image(
                image=im,
                caption='Saliency Map',
                use_column_width='always',
            )
        with c2:
            st.bokeh_chart(
                figure = make_barh(labels, scores),
                use_container_width=False
            )



if __name__ == "__main__":
    main()