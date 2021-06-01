FROM yolact-base:v0.1

ENV DEBIAN_FRONTEND noninteractive

ARG USERNAME=ros
USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo apt update \
    && sudo apt install -y --no-install-recommends --allow-unauthenticated \
        software-properties-common curl apache2-utils \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils curl ca-certificates gnupg patch \
        mesa-utils libgl1-mesa-dri ffmpeg tmux \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
        xvfb x11vnc libcanberra-gtk-module libcanberra-gtk3-module \
        vim-tiny sudo firefox-geckodriver ttf-ubuntu-font-family ttf-wqy-zenhei \
    && sudo add-apt-repository -r ppa:fcwu-tw/apps \
    && sudo mkdir /usr/local/ffmpeg \
    && sudo ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg \
    && sudo ln -s /usr/bin/python3 /usr/bin/python

ENV HOME=/home/ros \
    SHELL=/bin/bash

ENTRYPOINT ["/usr/bin/lxpanel", "--profile", "LXDE"]