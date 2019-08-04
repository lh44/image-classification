FROM tensorflow/tensorflow
LABEL maintainer="varunduttb@gmail.com"
RUN pip install Pillow && pip install -U scikit-learn
COPY src /app/
WORKDIR /app
CMD python Predict.py