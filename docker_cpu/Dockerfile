###############################################################################
# Build from the nightly tensorflow image. (root user)
###############################################################################
FROM tensorflow/tensorflow:2.8.0 AS tf28
# Set the working folder as root.
WORKDIR /root
# Install git, ssh, vim, and
# install graphviz, pydot-ng, and pydot, which are required to use the plot
# model function in tensorflow. Also, intall matplotlib. The argument -y is
# required for apt. Firstly apt update, and lastly rm -rf /var/lib/apt/lists/*
# in order to install the latest package and clean afterwards.
RUN apt update \
    && apt --no-install-recommends -y install git-core vim openssh-server graphviz wget apt-transport-https curl gnupg \
    && pip install --upgrade pip \
    autopep8 \
    black[jupyter] \
    pycodestyle \
    pydot-ng \
    pydot \
    graphviz \
    scipy \
    matplotlib \
    ipython \
    jupyter \
    pandas \
    sympy \
    nose \
    seaborn \
    scikit-learn \
    h5py \
    mlflow \
    "jax[cpu]" \
    git+https://github.com/deepmind/dm-haiku \
    optax \
    chex \
    dm-pix \
    absl-py \
    tqdm \
    thop \
    plotly \
    notebook \
    ipywidgets \
    && rm -rf /var/lib/apt/lists/* 
# docker build -t tfjax_cpu .
