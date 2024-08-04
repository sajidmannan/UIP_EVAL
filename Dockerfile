ARG PYTORCH_VERSION=2.3.1
ARG CUDA_VERSION=cuda12.1-cudnn8-devel
FROM pytorch/pytorch:${PYTORCH_VERSION}-${CUDA_VERSION}

RUN apt-get update -y && \
	apt-get upgrade -y && \
	apt-get install -y git make g++ gcc

# this arg can be used to forcibly re-clone and install
ARG BREAK_CACHE=1
COPY matsciml /opt/matsciml
WORKDIR /opt/matsciml
RUN pip install -f https://data.dgl.ai/wheels/repo.html -e './[all]'
# install newer CUDA build of DGL
RUN pip install dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
# newer versions of pymatgen can't deserialize currently saved LMDB
RUN pip install pymatgen==2023.9.25
# add extra packages needed here
RUN pip install tensorboardx wandb mlflow

# add generic user to prevent root
RUN groupadd user && useradd -d /home/user -g user user
RUN chown -R user:user /opt/matsciml
# make package directory also writeable
RUN chown -R user:user /opt/conda

USER user

HEALTHCHECK NONE
