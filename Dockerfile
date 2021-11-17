FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY unet /app/unet
COPY app.py unet_module.py image.jpg /app/

ARG SAGE_STORE_URL="https://osn.sagecontinuum.org"
ARG BUCKET_ID_MODEL="a42a4a21-f7ac-4a7c-ba76-325595637eee"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} CP_epoch228a.pth --target /app/wagglecloud_unet_300.pth

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
