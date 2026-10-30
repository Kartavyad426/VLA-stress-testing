# TaskGen chat in Docker (CPU)

A browser chat with the TaskGen agent. You describe a change ("add a three-drawer cabinet with the middle drawer
open and a red block inside; the robot puts it in the bowl"), and the agent edits the reconstructed MuJoCo scene and
writes the task. Every step shows up in the page: the model call, validation, each tool call. It runs on a laptop
CPU; no GPU is needed.

> **Setting this up with an AI agent?** Point it at [AGENTS.md](AGENTS.md): numbered steps with the exact commands,
> the expected output of each, a smoke test, and where to stop and ask you. It never needs to see your NIM key.

## Run

```bash
docker load -i taskgen_cpu.tar          # if you were given the image as a file (else: docker pull <registry>/taskgen:cpu)
docker run --rm -e NVIDIA_NIM_API_KEY=<your NIM key> -p 127.0.0.1:8765:8765 taskgen:cpu
```

Open http://127.0.0.1:8765. The NIM key is your own (https://build.nvidia.com). Without one, only the `fake` model
answers, with a fixed reply. To keep your chat sessions after the container stops, add
`-v taskgen_runs:/home/imerit/Documents/Code/VLA-exp-taskgen/runs`.

## What's inside

| | |
|---|---|
| Scenes | `orange`, `c11`, `c1b`, `c9` (res31's SceneGen v1 builds from DROID episodes), `orange_stereo` (the older hand-built one). All have composite assets. c1b's light fit and c9's gripper check failed res31's provisional checks, so their Composite view comes with a warning |
| Models | nemotron-3-ultra-550b (default), nemotron-3.5-lightning, laguna-xs-2.1, gpt-oss-20b, all via NIM; `fake` (offline) |
| Buttons | **Check**: does doing nothing fail, and can the scripted expert solve it (plain sim view). **Composite**: what a policy would see (the real background + sim, wrist view). **Run policy → expert**: the scripted expert in the policy's own cameras. **New chat**, **Undo**, **Reset**, **Save** |
| Not included | GR00T (GPU): the policy picker offers only *expert* |

Every turn is saved as a trace (sidebar → "trace"): the exact messages sent to the model, every tool call and its
result, and the scene and task configs.

## Known limits

- Chat state is per server, not per user: if two people open the same container, they share one scene and see each
  other's turns. Run one container per person.
- The agent can only add boxes (colour + size), a drawer, a 1–3 drawer cabinet, and objects from the kitchen and GSO
  asset packs. Base objects of the reconstructed scene can be moved by the task, not by the setup.
- Mac (Apple Silicon): this image is built for x86_64; Docker Desktop runs it under emulation, which should work but is slower and hasn't been tested. An arm64 build isn't done yet.

## Licences

DROID episodes: CC BY 4.0. Asset packs: NVIDIA Kitchen-MJCF (CC BY 4.0) and Google Scanned Objects (meshes
CC BY 4.0, XML MIT). MuJoCo Menagerie robot models (FR3, Robotiq 2F-85): see their folders. LaMa weights: Apache-2.0.
The two LIBERO-plus assets (black bowl, coffee cup): **licence not confirmed**, so iMerit-internal use only until it's checked.

Built from `experiments/taskgen/docker/` (`stage.sh`, then `docker build`). Exploratory, UNVERIFIED.
(session: TaskGen, 2026-10-30)
