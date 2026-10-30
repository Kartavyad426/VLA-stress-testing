# AGENTS.md: setting up the TaskGen chat (for an AI agent)

You are setting up the TaskGen chat for a person on their machine. TaskGen is a browser chat where an LLM agent edits
a reconstructed MuJoCo robot scene and writes tasks for it. Follow the steps in order. Each step has a command, the
result that means it worked, and what to do if it didn't. Stop and ask the person when a step says **ASK**.
The human-facing overview is README.md.

## Rules

1. **Never print, log, echo or commit the NIM API key.** Pass it only through the environment (`-e NVIDIA_NIM_API_KEY`).
   Don't write it into files, and don't paste it into command lines that get logged.
2. Publish the port on localhost only (`-p 127.0.0.1:8765:8765`), never `0.0.0.0`. The chat has no login, and it
   spends the person's NIM credits.
3. One container per person. The chat state is per server: two people on one container share a scene.
4. Don't commit or upload the image file (`taskgen_cpu.tar.gz`). It contains assets that are iMerit-internal only
   (README.md → Licences).

## Step 1: check the prerequisites

```bash
docker --version                      # expect: Docker version 2x.x or newer
docker info > /dev/null && echo OK    # expect: OK
df -h ~ | tail -1                     # expect: >= 20G available (image 8.5 GB + the load)
```

- `docker: command not found`: **ASK** the person to install Docker Desktop (Mac/Windows) or Docker Engine (Linux).
- `permission denied ... docker.sock` (Linux): **ASK** them to add their user to the `docker` group, or to run with `sudo`.
- Less than 20 GB free: **ASK** before continuing.
- No GPU is needed. If `uname -m` says `arm64` (Apple Silicon), go on, but use the `--platform linux/amd64` flag in
  step 4. The image is x86_64 only and runs emulated there (untested, slower).

## Step 2: get the image

From a file the person was given:

```bash
docker load -i taskgen_cpu.tar.gz     # expect: Loaded image: taskgen:cpu
```

(If the person was given a registry name instead: `docker pull <registry>/taskgen:cpu`, then
`docker tag <registry>/taskgen:cpu taskgen:cpu`.)

Verify:

```bash
docker images taskgen:cpu --format '{{.Repository}}:{{.Tag}} {{.Size}}'   # expect: taskgen:cpu 8.51GB
```

## Step 3: the NIM key

The agent's models run on NVIDIA NIM (https://build.nvidia.com). Check that the key is in the environment **without
printing it**:

```bash
[ -n "$NVIDIA_NIM_API_KEY" ] && echo "key is set" || echo "key is NOT set"
```

- `key is NOT set`: **ASK** the person to export it in their own terminal (`export NVIDIA_NIM_API_KEY=...`). Don't
  ask them to paste it to you. Without a key the chat still starts, but only the `fake` model answers (a fixed reply),
  so say this clearly if they want to go on without one.

## Step 4: start the container

```bash
docker run -d --rm --name taskgen \
  -e NVIDIA_NIM_API_KEY \
  -p 127.0.0.1:8765:8765 \
  -v taskgen_runs:/home/imerit/Documents/Code/VLA-exp-taskgen/runs \
  taskgen:cpu
```

`-e NVIDIA_NIM_API_KEY` with no `=value` passes the variable through without putting the key on the command line.
The volume keeps chat sessions and traces across restarts. On Apple Silicon, add `--platform linux/amd64` after `run`.

- `port is already allocated`: something uses 8765. Use `-p 127.0.0.1:8766:8765` and URL port 8766 from here on.
- `Conflict. The container name "/taskgen" is already in use`: run `docker rm -f taskgen`, then retry.

Wait until it's ready (up to 2 min):

```bash
timeout 120 sh -c 'until docker logs taskgen 2>&1 | grep -q "TaskGen chat on"; do sleep 2; done' && echo READY
```

- No `READY`: run `docker logs taskgen 2>&1 | tail -30` (it never contains the key) and look for a Python traceback.
  Report it to the person verbatim.

## Step 5: smoke test

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8765/        # expect: HTTP 200

docker exec taskgen /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/python -c '
import asyncio, json, websockets
async def m():
    async with websockets.connect("ws://127.0.0.1:8765/ws", max_size=None) as w:
        h = json.loads(await w.recv())
        assert h["type"] == "hello" and "orange" in h["scenes"], h
        await w.send(json.dumps({"type": "render"}))
        while True:
            r = json.loads(await asyncio.wait_for(w.recv(), 120))
            if r["type"] == "render": print("SMOKE OK: scenes", h["scenes"], "| images", list(r["images"])); return
asyncio.run(m())'
# expect: SMOKE OK: scenes ['orange', 'c11', 'c1b', 'c9', 'orange_stereo'] | images ['ext', 'wrist', 'free']
```

This test makes no model call (no NIM credits used). It checks that the server answers and that MuJoCo renders.
Tested on the image, 2026-10-30.

## Step 6: hand over

Tell the person:

- Open **http://127.0.0.1:8765** in a browser.
- Pick a scene and a model at the top, then type a request, e.g. *"add a red block next to the bowl; the robot
  puts it in the bowl"*. Every step the agent takes appears as a card, and you can click one to see its input and result.
- **Check** tests the task: doing nothing must fail and the scripted expert must solve it. **Composite** shows what a
  policy would see. The sidebar lists the session's turns; "trace" opens the full record of a turn.
- The first model reply takes about 10–60 s, depending on the model.

## Stop and clean up

```bash
docker stop taskgen                      # the chat stops; sessions stay in the taskgen_runs volume
docker volume rm taskgen_runs            # only if the person wants their sessions deleted (ASK first)
docker rmi taskgen:cpu                   # only to free the 8.5 GB (ASK first)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The chat says `NVIDIA_NIM_API_KEY not set` | the container was started without the key | step 3, then `docker rm -f taskgen` and step 4 |
| A model card ends in `ReadTimeout` | NIM is slow or overloaded for that model | retry, or pick another model (nemotron-3-ultra-550b was the most reliable on 10-30) |
| "No composite scene available" | that scene's composite assets are missing | press **Yes, build it** (~8–10 min, CPU) |
| Composite shows a ⚠ warning (c1b, c9) | res31's provisional checks failed for that scene | the view may be off; use orange or c11 for policy-view questions |
| Check fails with the expert not grasping | the block was placed at the edge of reach (known, L57) | ask for a placement nearer the robot |
| The page says "disconnected" | the container stopped | `docker logs taskgen`, then step 4 again |

Exploratory, UNVERIFIED. (session: TaskGen, 2026-10-30)
