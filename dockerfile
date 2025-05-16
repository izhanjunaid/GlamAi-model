FROM continuumio/miniconda3:4.12.0

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "elegant", "/bin/bash", "-c"]

COPY . /app
COPY ckpts/G.pth /app/ckpts/
COPY ckpts/sow_pyramid_a5_e3d2_remapped.pth /app/ckpts/
COPY faceutils/dlibutils/shape_predictor_68_face_landmarks.dat /app/faceutils/dlibutils/

EXPOSE 8000

CMD ["conda", "run", "-n", "elegant", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]