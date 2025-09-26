FROM mambaorg/micromamba:2.3-ubuntu22.04
LABEL authors="Gisela Gabernet" \
    description="Docker image containing all requirements for AMULETY"

COPY environment.yml /
RUN micromamba create -n amulety -c conda-forge -c bioconda python=3.8 igblast=1.22.0 pip
RUN micromamba clean -a
ENV PATH /opt/conda/envs/amulety/bin:$PATH

RUN igblastn -version
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Add the source files to the image
COPY . /tmp
WORKDIR /tmp

# Install current
USER root
RUN rm -rf *.egg-info
RUN python -m pip install gensim==3.8.3
RUN python -m pip install .
RUN amulety --help

# Install git to clone Immune2Vec repository
RUN apt-get update && apt-get install -y git
# Clone Immune2Vec repository
RUN git clone https://bitbucket.org/yaarilab/immune2vec_model.git
ENV IMMUNE2VEC_PATH /tmp/immune2vec_model
ENV AMULETY_CACHE /tmp/amulety_cache
USER $MAMBA_USER
