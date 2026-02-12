# LLMs Don't Keep Secrets

**Demonstration of transcript-conditioned reasoning in large language models**

When a language model plays 20 Questions as the secret-holder, does it commit to a secret internally and answer from that hidden state? Or does it simply reconstruct the most likely secret from the visible transcript each time it generates a token?

This experiment shows it's the latter.

## The Experiment

We define a small secret space of six animals (cat, dog, eagle, salmon, shark, crocodile), each described by 8 binary features. For each animal, we synthetically generate a Q&A transcript where the answers uniquely identify that animal by elimination. We then feed each transcript to the same model instance and ask it to reveal "its" secret.

**The model was never part of the original conversation** — so whatever it declares, it's reconstructing from the transcript.

### Results (Llama-3.1-70B-Instruct)

| Transcript implies | Model declared | Match |
|--------------------|----------------|-------|
| cat                | cat            | ✓     |
| dog                | dog            | ✓     |
| eagle              | eagle          | ✓     |
| salmon             | salmon         | ✓     |
| shark              | shark          | ✓     |
| crocodile          | crocodile      | ✓     |

**6/6 correct.** Different transcripts → different declared secrets, from the same model instance with identical weights and temperature.

### The Contradictory Case

We also tested a logically impossible transcript (mammal = Yes, fish = Yes, flies = Yes). No animal in the secret space matches. The model recognized the contradiction and admitted error — rather than revealing a committed secret, because it never had one.

## Interactive Demo

Open [`transcript_bubbles.html`](transcript_bubbles.html) in a browser to see the experiment visualized as a chat interface.

> To serve it via GitHub Pages, enable Pages in your repo settings and point it to the root of the `main` branch. The demo will be available at `https://<username>.github.io/<repo>/transcript_bubbles.html`.

## Running the Notebook

### Requirements

- Python 3.10+
- GPU with ≥40GB VRAM (tested on NVIDIA GH200 96GB)
- HuggingFace account with Llama 3.1 license accepted

### Setup

```bash
pip install torch transformers accelerate bitsandbytes huggingface_hub
huggingface-cli login
```

### Run

Open `transcript_conditioning_demo.ipynb` and run all cells top-to-bottom. The notebook loads Llama-3.1-70B-Instruct in 4-bit quantization, generates transcripts for all six secrets, runs inference, and prints results — including the contradictory case.

## What This Means

Without external memory infrastructure, an LLM playing 20 Questions is **transcript-conditioned** — it reconstructs its "secret" from the conversation history at every generation step. It doesn't maintain hidden state; it performs inference over visible text. This has implications for how we design games, evaluations, and agents that assume models can hold private information.
