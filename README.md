# Computer Vision Flower Classification with Resnet50

![](./preview/preview-1.gif)

## Computer Vision Flower Classification with Resnet50, TensorFlow and Streamlit

The Face Mask Detection Model was trained using the YOLOv8 Architecture, the Google Colab GPU and with over 2600 Face Mask Images. The U.I. was built with Streamlit. It can be test it with uploading a Picture, Video, or in Real Time Detection.

## Run it Locally

Test it Locally by running the `app.py` file, built with `Streamlit`, and the `api.py` file with `Flask`. Remember first to run the `api.py` file, copy the http url and saved in the API variable of the `app.py` file, and uncomment the code lines.

## App made with Streamlit
```sh
streamlit run app.py
```

## Deployed with Flask
```sh
python3 api.py
```

![](./preview/preview-2.gif)

## Resources
Flowers Dataset: https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset?select=dataset
