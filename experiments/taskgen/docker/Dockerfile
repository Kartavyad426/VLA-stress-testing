# TaskGen chat (agent: scene modification + task creation; Check with the scripted expert; Composite; expert rollouts).
# CPU only, no GPU needed. Build: experiments/taskgen/docker/stage.sh, then docker build -t taskgen:cpu <ctx>.
# Run:  docker run --rm -e NVIDIA_NIM_API_KEY=... -p 127.0.0.1:8765:8765 taskgen:cpu   -> http://127.0.0.1:8765
# Exploratory, UNVERIFIED. (session: TaskGen, 2026-10-30)
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 procps \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONDONTWRITEBYTECODE=1 TASKGEN_POLICIES=expert
# the venv at the path the CPU wrapper (py_triton_first.sh) expects
COPY requirements-cpu.txt /tmp/
RUN python -m venv /home/imerit/Documents/Code/VLA/.venvs/real2sim \
    && /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/pip install --no-cache-dir -r /tmp/requirements-cpu.txt
COPY root/ /
COPY entrypoint.sh /entrypoint.sh
EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
