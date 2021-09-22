FROM gcr.io/deeplearning-platform-release/base-cpu:latest as intermediate
# Add metadata identifying these images as our build containers (this will be useful later!)
LABEL stage=intermediate

RUN git clone --branch for_kubeflow https://a5d283bca2fd735e3e6d27d2b0e9106abe9a83c0@github.com/tourist-C/HER2
# set base image (host OS)
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-4
#gcr.io/deeplearning-platform-release/tf2-gpu.2-3

# # needed for shaply 
# # https://www.edureka.co/community/12751/how-to-install-packages-inside-a-docker-ubuntu-image
# # https://nickjanetakis.com/blog/docker-tip-17-breaking-up-long-lines-in-your-dockerfile
# # add -y flag to prevent script stopping when it asks you to press Y to confirm
# # -qq No output except for errors
# # concat apt-get command under 1 "RUN" command to that everything is done in 1 docker layer
# # https://stackoverflow.com/questions/11245865/java-home-does-not-point-to-the-jdk
RUN apt-get update -y \
    && apt-get install -y python3-openslide ffmpeg libsm6 libxext6 # linux openslide installation is different from windows

# # install dependencies
RUN pip install -U openslide-python shapely matplotlib pathos opencv-python geojson

COPY ./data_gen.py /ae13/tile_engine/
COPY ./mass_data_gen.py /ae13/tile_engine/
COPY ./components/tile_engine_component/src /ae13/tile_engine

COPY --from=intermediate /HER2/scancer/src/ ae13/tile_engine/src/

# copy the content of the local directory to the working directory
#COPY ./src/tile_engine_component /ae13/

RUN rm -rf ~/.matplotlib\ 
    rm -rf ~/.cache/matplotlib\
    rm -rf ~/.cache/fontconfig/
