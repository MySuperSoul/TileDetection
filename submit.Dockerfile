FROM registry.cn-hangzhou.aliyuncs.com/tile/tile_submit:mmdet-mish-v4

# RUN pip install ensemble_boxes -i https://mirrors.aliyun.com/pypi/simple
ADD . /work
WORKDIR /work

CMD ["sh", "run.sh"]