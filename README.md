# Integrated Deep Learning and GRAD- CAM Visualisation for [Skin Cancer Classification Using Streamlit - Framework](https://www.kaggle.com/api/v1/datasets/download/fanconic/skin-cancer-malignant-vs-benign)


## Objective
This repository aims to integrate the ML models with GRAD-CAM visualization tool to inhance the interpredibility.

## Code Structure / Services
- `notebooks` - Contains the codes for the ML models including all the transformation while selecting the backbone, building the model.

- `src` - Contains frontend services along with the images and models.
    - `img_examples` - Malignant and Benign images to check their probability.
    - `models` - VIT and Mobilenet model.
    - `utils` - It contains the prepossing and grad-cam visulization technique for both VIT and Mobilenet model.
    - `Dockerfile` - Contain set of commands to assemble the docker images.
    - `requirements.txt` - List of packages and libraries needed to work on this project.
    - `str_app.py` - Basic frontend app developed using [Streamlit](https://streamlit.io/).

- `docker-compose` - Compose file starts frontend services to run application by going into source(str_app).

## Deployment
- Local deployment
    - Install Docker. Instructions available [here](https://docs.docker.com/engine/install/). Make sure docker is up and running before proceeding.
    - Install Git. Instruction [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
    - Clone repo and run compose
    ```
    git clone https://github.com/Tulika-Yadav/ML-ASSESMENT.git && cd ./ML-ASSESMENT
    docker compose up
    ```
    