#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

#Python Versions
#FROM python:3
#Python Anaconda
FROM continuumio/anaconda3

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. It is suggested that you only do this if your
# requirements rarely (if ever) change.
# COPY requirements.txt /tmp/pip-tmp/

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git openssh-client less iproute2 procps lsb-release gcc g++ python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev\
    #
    # Install DataScience Python Libraries
    && pip --disable-pip-version-check --no-cache-dir install pylint \ 
    && pip --disable-pip-version-check --no-cache-dir install jupyter \
    && pip --disable-pip-version-check --no-cache-dir install numpy \
    && pip --disable-pip-version-check --no-cache-dir install pandas \
    && pip --disable-pip-version-check --no-cache-dir install datatable \
    && pip --disable-pip-version-check --no-cache-dir install scipy \
    && pip --disable-pip-version-check --no-cache-dir install folium==0.2.1 \
    && pip --disable-pip-version-check --no-cache-dir install matplotlib \
    && pip --disable-pip-version-check --no-cache-dir install ipywidgets>=7.0.0 \
    && pip --disable-pip-version-check --no-cache-dir install bqplot \
    && pip --disable-pip-version-check --no-cache-dir install nbinteract==0.0.12 \
    && pip --disable-pip-version-check --no-cache-dir install pycaret \
    && pip --disable-pip-version-check --no-cache-dir install xgboost \
    && pip --disable-pip-version-check --no-cache-dir install lightgbm \
    && pip --disable-pip-version-check --no-cache-dir install catboost \
    && pip --disable-pip-version-check --no-cache-dir install geppy \
    && pip --disable-pip-version-check --no-cache-dir install gplearn \
    && pip --disable-pip-version-check --no-cache-dir install mlxtend \
    && pip --disable-pip-version-check --no-cache-dir install mlens \
    && pip --disable-pip-version-check --no-cache-dir install scikit-optimize \
    && pip --disable-pip-version-check --no-cache-dir install sklearn-genetic \
    && pip --disable-pip-version-check --no-cache-dir install shap \
    && pip --disable-pip-version-check --no-cache-dir install eli5 \
    && pip --disable-pip-version-check --no-cache-dir install pandarallel \
    && pip --disable-pip-version-check --no-cache-dir install swifter \
    && git clone https://github.com/hyperopt/hyperopt-sklearn.git && cd hyperopt-sklearn && pip install --disable-pip-version-check --no-cache-dir -e . && cd .. && rm -vr hyperopt-sklearn \
    #
    # Update Python environment based on requirements.txt
    ## && pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    ## && rm -rf /tmp/pip-tmp \
    #
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog