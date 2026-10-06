import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio.to_thread
import logging_db
import numpy as np
import onnxruntime as ort
import redis.asyncio as redis
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import preprocess, softmax
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "model_fp32.onnx"
BATCH_TIMEOUT = 0.002  # 2ms window, starting from the first request in a new batch
BATCH_MAX_SIZE = 4
CACHE_TTL_SECONDS = 60

# "localhost" for running the backend directly on the host; docker-compose
# overrides this to "redis" so it resolves to the compose service by name.
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# Loaded once, here at import time, not per-request or via a lifespan hook --
# there's no async setup needed, so eager module-level init is simplest way
# to get "one warm session shared by every request" (same goal as a lifespan
# handler, just without the extra ceremony since nothing here needs await).
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

app = FastAPI(title="MNIST Digit Recognizer API (ONNX, dynamic batching)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_api_request(request: Request, call_next):
    # Records one api_requests row per /predict call, success OR failure.
    # This is what makes a real error rate possible: a 400 for a bad upload
    # never reaches the model, so the predictions table can't see it. status
    # starts at 500 so an exception that escapes call_next is logged as a
    # server error (and still propagates to the client as one).
    start = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if request.url.path == "/predict":
            await anyio.to_thread.run_sync(
                partial(
                    logging_db.log_api_request,
                    endpoint=request.url.path,
                    status_code=status,
                    latency_ms=(time.time() - start) * 1000,
                )
            )


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


# Pairs one request's input with a private Future it can await -- this is
# what lets many concurrent requests share one batched inference call while
# each still only ever sees its own result, never anyone else's.
@dataclass
class BatchItem:
    tensor: np.ndarray
    future: "asyncio.Future[np.ndarray]"


# The single hand-off point between predict() (producer) and batch_worker()
# (sole consumer) -- requests don't call the model directly, they queue here.
# Created at startup (not import) so it belongs to the event loop that actually
# serves requests; a queue made at import binds to whichever loop touches it first.
request_queue: "asyncio.Queue[BatchItem]" = None  # type: ignore[assignment]


def run_batch_inference(batch_tensor: np.ndarray) -> np.ndarray:
    # Runs on a worker thread (see anyio.to_thread.run_sync below), not the
    # event loop thread, so a slow batch still doesn't block new requests
    # from queuing up for the *next* batch while this one runs.
    infer_start = time.time()
    logits_batch = session.run(["logits"], {"input": batch_tensor})[0]
    infer_end = time.time()
    print(
        f"[batch] size={batch_tensor.shape[0]} infer_start={infer_start:.4f} "
        f"infer_end={infer_end:.4f} duration={infer_end - infer_start:.4f}s",
        flush=True,
    )
    return logits_batch


async def batch_worker():
    # Single consumer, so there's no real race to lock against -- asyncio is
    # cooperative, and only one coroutine ever runs between await points.
    loop = asyncio.get_event_loop()
    while True:
        first_item = await request_queue.get()
        batch = [first_item]
        # Deadline is fixed from this first item, not reset per arrival, so
        # a steady trickle of requests can't stall a batch indefinitely.
        deadline = loop.time() + BATCH_TIMEOUT

        while len(batch) < BATCH_MAX_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                # Whichever hits first -- BATCH_MAX_SIZE items, or the
                # shrinking deadline -- ends the wait; that's the whole
                # size-vs-timeout race, expressed as a shrinking wait_for.
                batch.append(await asyncio.wait_for(request_queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break

        batch_tensor = np.concatenate([item.tensor for item in batch], axis=0)
        logits_batch = await anyio.to_thread.run_sync(run_batch_inference, batch_tensor)

        # Concatenation preserves arrival order, so index i into the batch
        # list and row i of the output always belong to the same request --
        # this is what routes each result back to the request that's waiting
        # on it, not just whichever one happens to wake up first.
        for i, item in enumerate(batch):
            item.future.set_result(logits_batch[i])


@app.on_event("startup")
async def launch_batch_worker():
    global request_queue
    request_queue = asyncio.Queue()
    asyncio.create_task(batch_worker())


@app.on_event("startup")
async def setup_logging_db():
    # init_db() only runs once at startup, not per-request, so a plain
    # blocking call here is fine -- it's not competing with live traffic
    # for the event loop the way a per-request DB write would be.
    logging_db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/reload-model")
async def reload_model():
    # Why this has to exist at all: `session` was created ONCE, at import
    # time (see above), inside this one running process. The Week 8
    # retraining pipeline runs as a SEPARATE process/script -- when it
    # overwrites model_fp32.onnx on disk with a new, better-performing
    # candidate, that write is invisible to this already-running process
    # until something tells it to re-read the file. Swapping the file
    # alone is NOT the same as redeploying it; this endpoint is the "tell
    # it" step that makes the swap actually take effect.
    #
    # No auth here for project scope -- in a real deployment this would
    # need to be restricted (internal network only, a shared secret, etc.)
    # since anyone who can call it can swap the live model.
    global session
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    # Every cached entry in Redis was computed by the model we just
    # replaced. Without this, a repeat of an already-cached image would
    # keep returning the OLD model's answer, silently, until its 60s TTL
    # happens to expire -- discovered by actually testing the reload, not
    # by guessing: reload alone looked like it worked until a *fresh*
    # image was used to prove it, which is what exposed this.
    cache_keys = [key async for key in redis_client.scan_iter(match="predict:*")]
    if cache_keys:
        await redis_client.delete(*cache_keys)

    return {"status": "reloaded", "model_path": str(MODEL_PATH), "cache_entries_flushed": len(cache_keys)}


@app.get("/debug/sleep")
async def debug_sleep():
    # Temporary diagnostic: a pure Python CPU-bound blocking call with no
    # I/O, no numpy, no C-extension. Isolates whether "sync code inside an
    # async def" truly serializes requests, independent of onnxruntime.
    time.sleep(0.5)
    return {"status": "done"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    # Optional, client-set tag -- NOT part of the model's decision in any
    # way (prediction logic below never reads it). Its only job is telling
    # the logging layer who sent this request, so the dashboard can tell
    # "the workload generator's synthetic/drift traffic" apart from real
    # traffic rather than silently mixing them into one population.
    x_traffic_source: str = Header(default="live", alias="X-Traffic-Source"),
    # Optional. Set ONLY by the workload generator, which knows the real
    # MNIST label of the image it's sending. Logged purely so the Week 8
    # retraining pipeline has labeled data to fine-tune on later -- it is
    # never read anywhere in the prediction path above, so it cannot
    # influence the model's actual answer. A real user's request never
    # sets this; main.py has no way to know their true digit.
    x_true_label: str | None = Header(default=None, alias="X-True-Label"),
):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Validate the optional label up front: a bad header should be a clean
    # 400, not a 500 after the model already ran and the result was cached.
    true_label: int | None = None
    if x_true_label is not None:
        if not x_true_label.isdigit() or not 0 <= int(x_true_label) <= 9:
            raise HTTPException(status_code=400, detail="X-True-Label must be a digit 0-9.")
        true_label = int(x_true_label)

    request_start = time.time()
    image_bytes = await file.read()

    # Hashing the raw bytes (not the preprocessed tensor) and checking before
    # preprocess() runs means a hit skips decode + batching + inference
    # entirely, not just the model call -- this has to come first to matter.
    cache_key = "predict:" + hashlib.sha256(image_bytes).hexdigest()
    cached = await redis_client.get(cache_key)
    if cached is not None:
        print(f"[cache] HIT  key={cache_key[:20]}... lookup={time.time() - request_start:.4f}s", flush=True)
        response = PredictionResponse(**json.loads(cached))
        # On a hit we deliberately do NOT run preprocess() -- that's the
        # whole point of checking the cache first. So there's no tensor to
        # take pixel stats from here; we log pixel_mean/std as NULL rather
        # than paying the decode cost just to populate a log column. Drift
        # detection (Week 7 dashboard) reads from the miss population, which
        # is fine since exact-repeat hits are rare for this domain (unique
        # hand-drawn digits), per the Week 6 caching discussion.
        await anyio.to_thread.run_sync(
            partial(
                logging_db.log_prediction,
                digit=response.digit,
                confidence=response.confidence,
                latency_ms=(time.time() - request_start) * 1000,
                pixel_mean=None,
                pixel_std=None,
                cache_hit=True,
                source=x_traffic_source,
            )
        )
        return response
    print(f"[cache] MISS key={cache_key[:20]}...", flush=True)

    try:
        tensor = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    # Hand off to batch_worker via the queue, then suspend on our own Future
    # -- awaiting an unresolved Future frees the event loop for other
    # requests instead of blocking, and only resumes once batch_worker
    # resolves this specific Future with this request's own result.
    future: asyncio.Future[np.ndarray] = asyncio.get_event_loop().create_future()
    await request_queue.put(BatchItem(tensor=tensor, future=future))
    logits = await future

    probabilities = softmax(logits)
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    response = PredictionResponse(
        digit=digit,
        confidence=confidence,
        probabilities=[float(p) for p in probabilities.tolist()],
    )
    await redis_client.set(cache_key, response.model_dump_json(), ex=CACHE_TTL_SECONDS)
    print(f"[cache] computed and stored in {time.time() - request_start:.4f}s (ttl={CACHE_TTL_SECONDS}s)", flush=True)

    # tensor is the normalized (1,1,28,28) array straight from preprocess(),
    # the exact same values the model saw -- so pixel_mean/std here describe
    # "what the model is actually being asked to classify right now," which
    # is what we want to compare against a training-time baseline later.
    await anyio.to_thread.run_sync(
        partial(
            logging_db.log_prediction,
            digit=digit,
            confidence=confidence,
            latency_ms=(time.time() - request_start) * 1000,
            pixel_mean=float(tensor.mean()),
            pixel_std=float(tensor.std()),
            cache_hit=False,
            source=x_traffic_source,
            true_label=true_label,
            # Store only when we have a label to go with it -- see the
            # note on logging_db.log_prediction for why.
            image_bytes=image_bytes if true_label is not None else None,
        )
    )
    return response
