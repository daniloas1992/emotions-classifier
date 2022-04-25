FROM tensorflow/tensorflow:latest-gpu

USER root

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ADD . /app
WORKDIR /app
EXPOSE 3000

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install np
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install Keras
RUN pip3 install silence_tensorflow
RUN pip3 install opencv-python

CMD ["sh"]





