# set base image (host OS)
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-4
#gcr.io/deeplearning-platform-release/tf2-gpu.2-3

# RUN apt-get update -y \
#     && apt-get install -y python3-openslide ffmpeg libsm6 libxext6 # linux openslide installation is different from windows

# # install dependencies
RUN pip install -U shapely matplotlib pathos opencv-python geojson scikit-image


COPY ./src /ae13/evaluate

# copy the content of the local directory to the working directory
#COPY ./src/tile_engine_component /ae13/

RUN rm -rf ~/.matplotlib\ 
    rm -rf ~/.cache/matplotlib\
    rm -rf ~/.cache/fontconfig/
