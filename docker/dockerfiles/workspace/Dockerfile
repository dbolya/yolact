FROM yolact-base:v0.1

# Create workspace
ARG USERNAME=ros
USER $USERNAME
WORKDIR /home/$USERNAME

RUN mkdir -p /home/$USERNAME/workspace/src/yolact
RUN git clone https://github.com/cmrobotics/yolact.git /home/$USERNAME/workspace/src/yolact
RUN git clone https://github.com/tteepe/DCNv2 /home/$USERNAME/workspace/src/DCNv2
RUN git clone https://github.com/jinfagang/DCNv2_latest.git /home/$USERNAME/workspace/src/DCNv2_latest
#RUN rm -rf /home/$USERNAME/workspace/src/yolact/external/DCNv2/src
#RUN cp -r ~/workspace/src/DCNv2/DCN/src/* ~/workspace/src/yolact/external/DCNv2/src
#RUN cd ~/workspace/src/yolact/external/DCNv2 &&  python setup.py install