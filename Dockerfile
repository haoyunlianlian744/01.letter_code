# Wrought Aluminum Alloy ML Pipeline — Docker Environment
#
# Build:
#   docker build -t al-alloy-ml .
#
# Run (interactive Jupyter):
#   docker run -p 8888:8888 -v "$(pwd)":/workspace al-alloy-ml
#
# Run (execute full pipeline):
#   docker run -v "$(pwd)":/workspace al-alloy-ml \
#     jupyter nbconvert --to notebook --execute --inplace code/*.ipynb

FROM python:3.11-slim

LABEL description="Wrought Al alloy ML pipeline: feature engineering, SHAP, inverse design"
LABEL maintainer="CX"

# System dependencies for pymatgen + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Jupyter config
RUN mkdir -p /root/.jupyter
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
