FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

LABEL com.nvidia.volumes.needed=nvidia_driver

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 --timeout=300 --no-cache-dir install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install transformers==4.26.0 sentencepiece accelerate==0.15.0 bitsandbytes-cuda113

COPY ./app /app
COPY ./flan-t5-xxl-sharded-fp16/ /flan-t5-xxl-sharded-fp16

EXPOSE 7080

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:7080", "main:app","--workers","1","--timeout","180","-k","uvicorn.workers.UvicornWorker"]
