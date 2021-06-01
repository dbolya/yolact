FROM yolact-base:v0.1

# Create workspace
ARG USERNAME=ros
USER $USERNAME
WORKDIR /home/$USERNAME

RUN mkdir -p /home/$USERNAME/workspace/src/yolact
RUN git clone https://github.com/cmrobotics/yolact.git /home/$USERNAME/workspace/src/yolact
RUN git clone https://github.com/jinfagang/DCNv2_latest.git /home/$USERNAME/workspace/src/DCNv2_latest
RUN sudo chown -R ros:ros /opt/conda
RUN cd /home/$USERNAME/workspace/src/DCNv2_latest && python setup.py build develop