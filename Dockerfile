# StudyMate on Linux, built from the same environment.yml as the local conda env.
# Run it with compose (it adds the GPU, the models and the data folders): docker compose up --build
FROM mambaorg/micromamba:2.9.0-debian13-slim

# opencv-python needs these system libraries on Linux; the app folders must be writable by the app user
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/Data /app/uploads /hf && chown -R $MAMBA_USER:$MAMBA_USER /app /hf
USER $MAMBA_USER

# Dependencies before code, so editing code doesn't reinstall torch
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
COPY --chown=$MAMBA_USER:$MAMBA_USER server ./server
COPY --chown=$MAMBA_USER:$MAMBA_USER web ./web

# 0.0.0.0: inside a container 127.0.0.1 can't be reached from Windows
ENV STUDYMATE_HOST=0.0.0.0 STUDYMATE_PORT=5000 HF_HOME=/hf
EXPOSE 5000
CMD ["python", "server/http_server.py"]
