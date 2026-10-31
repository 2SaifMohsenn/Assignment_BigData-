FROM quay.io/jupyter/scipy-notebook:latest

WORKDIR /home/jovyan/work/bd-a1

COPY --chown=${NB_UID}:${NB_GID} requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=${NB_UID}:${NB_GID} analysis.py ./analysis.py
COPY --chown=${NB_UID}:${NB_GID} data/dataset.csv ./data/dataset.csv
COPY --chown=${NB_UID}:${NB_GID} notebook/ ./notebook/
COPY --chown=${NB_UID}:${NB_GID} output/ ./output/

EXPOSE 8888

CMD ["start-notebook.py", "--ServerApp.root_dir=/home/jovyan/work/bd-a1"]
