ARG CUDA="11.1"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu20.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update -y

RUN apt-get update -y \
 && apt-get install -y apt-utils git curl wget ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& chmod +x /Miniconda3-latest-Linux-x86_64.sh \
&& /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
&& rm /Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name e8 python=3.8.12 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=e8
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm

ARG CUDA
RUN conda install pytorch torchvision torchaudio cudatoolkit=${CUDA} -c pytorch -c nvidia -y\
&& conda clean -ya

# Install TorchVision master
# RUN git clone https://github.com/pytorch/vision.git \
#  && cd vision \
#  && python setup.py install

# install pycocotools
# RUN git clone https://github.com/cocodataset/cocoapi.git \
#  && cd cocoapi/PythonAPI \
#  && python setup.py build_ext install

RUN git clone https://github.com/crazyboy9103/e8ight 

WORKDIR /e8ight
