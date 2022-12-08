###############################################################################
# Build from the nightly tensorflow image. (root user)
###############################################################################
FROM tensorflow/tensorflow:2.8.0-gpu AS tf28
# Set the working folder as root.
WORKDIR /root
# Install git, ssh, vim, and
# install graphviz, pydot-ng, and pydot, which are required to use the plot
# model function in tensorflow. Also, intall matplotlib. The argument -y is
# required for apt. Firstly apt update, and lastly rm -rf /var/lib/apt/lists/*
# in order to install the latest package and clean afterwards.
RUN apt update \
    && apt --no-install-recommends -y install git-core vim openssh-server graphviz wget apt-transport-https curl gnupg \
    && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg \
    && mv bazel.gpg /etc/apt/trusted.gpg.d/ \
    && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt update && apt install bazel \
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
    jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    git+https://github.com/deepmind/dm-haiku \
    optax \
    chex \
    dm-pix \
    absl-py \
    tqdm \
    thop \
    plotly==4.14.3 notebook>=5.3 ipywidgets>=7.5 \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    && apt update && apt full-upgrade -y \
    && rm -rf /var/lib/apt/lists/* 
# docker build -t tfjax .
