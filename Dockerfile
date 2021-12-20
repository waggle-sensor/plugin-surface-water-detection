FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

RUN apt-get update \
  && apt-get install -y \
  build-essential \
  python3-dev \
  libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed PyYAML
RUN pip3 install --no-cache-dir --force -r /app/requirements.txt

COPY libs /app/libs
COPY configs /app/configs
COPY data /app/data
COPY app.py convert.py demo.py hubconf.py main.py /app/

ARG SAGE_STORE_URL="https://osn.sagecontinuum.org"
ARG BUCKET_ID_MODEL="3562bef2-735b-4a98-8b13-2206644bdb8e"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} deeplabv2_resnet101_msc-cocostuff164k-100000.pth --target /app/deeplabv2_resnet101_msc-cocostuff164k-100000.pth

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
