FROM tensorflow/tensorflow
LABEL maintainer="varunduttb@gmail.com"
RUN pip install -U Pillow==6.1.0 && pip install -U scikit-learn==0.20.4 && pip install -U Flask==0.10.1
ENV NUM_OF_CLASSES=4,
COPY src /app/
WORKDIR /app
EXPOSE 3000
CMD python Predict.py