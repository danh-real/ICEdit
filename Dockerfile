FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git ca-certificates && apt-get clean
COPY requirements.txt /tmp
COPY train/requirements.txt /tmp/train_requirements.txt
RUN pip install -r /tmp/requirements.txt && pip install --upgrade google-cloud-storage
RUN pip install -r /tmp/train_requirements.txt
RUN apt-get install -y tmux && apt-get install -y libevent-dev ncurses-dev build-essential bison pkg-config
