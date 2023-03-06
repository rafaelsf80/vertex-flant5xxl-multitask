FROM python:3.7

RUN pip install gradio google-cloud-aiplatform

COPY ./app /app

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]