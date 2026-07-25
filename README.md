# Learning World Models with DreamerV3 and R2-Dreamer

## Context

For anyone landing on this repository, I'm Guillermo, Data Scientist, and I've spent most of my career building **Recommender Systems** and **Rankings**. 

Lately, however, I've been more interested in what people call **physical AI**. My belief is that as AI gets more capable, the next frontier is to make it act in the real world, I mean **physically**. There is this viral quote that I think captures the idea:

<p align="center"><i>"I want AI to do my laundry and dishes so that I can do art and writing,<br>not for AI to do my art and writing so that I can do my laundry and dishes."</i><br>(<a href="https://x.com/AuthorJMac/status/1773679197631701238">Joanna Maciejewska</a>)</p>

The problem is that I have no experience in those fields (for example, I have never used Reinforcement Learning), and I have no idea how to get started.

To kick things off, I have read a bunch of blogposts, YouTube videos, and papers (check the [References](#references) section), and one that caught my attention was [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). This paper describes `DreamerV3`, a World Model agent that learns to act in a variety of environments by first learning a model of how the world behaves. This paper, later led me to a second one, [R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation](https://openreview.net/forum?id=Je2QqXrcQq), which introduces a related method that changes how the World Model learns its visual representation.

## Objective

This repository will be a summary of my personal study to understand the DreamerV3 and R2-Dreamer papers. I plan to run a comparison of the two methods (original `DreamerV3` vs. the newer `R2-Dreamer`) on a simple visual task. I will document my progress here, and I hope **this could be useful for anyone else trying to get started in this field**. If it is, please feel free to reach out to me, for example, via [Linkedin](https://www.linkedin.com/in/guillermo-lahuerta/).

## Why DreamerV3

I have been reading a lot about Physical AI, and the main idea that caught my attention was the following: what if instead of learning to act directly from real experience, we could build a system where **the agent first learns a model of how its world behaves**, and then uses that model to **train itself on its imagination**. That is essentially what a "**World Model**" allows you to do.

From what I have read online, `DreamerV3` is a great place to start. It compresses what it sees into a **compact latent state**, learns to predict how that state evolves when it acts, and trains its policy inside those predictions, on imagined futures! The agent basically learns from its own "dreams".

One of the main achievements of this model is that it was the first agent that collected diamonds in Minecraft without any human data (I have never played Minecraft, but I found it funny that the top minds in AI use it as a benchmark).

For my study, however, I will use [NM512/r2dreamer](https://github.com/NM512/r2dreamer), the codebase released with the separate R2-Dreamer paper. This was convenient because it already implements the DreamerV3 baseline, and it is written in PyTorch, which I am more familiar with than JAX.

## How to follow along

I will update this repo as I go with what I learned, what worked and what did not. The log below is kind of a **diary of the project**.

Note that **I do not commit to a fixed schedule**, so updates will be irregular. Basically I plan to work on this whenever I have free time.

---

## Log

### July 5th, 2026

I have started by creating this repo and writing this initial README to summarize this project (which for sure will evolve over time). My next step is to read the DreamerV3 paper to understand the model, and separately inspect `r2dreamer`, the codebase from the R2-Dreamer paper.

### July 25th, 2026

The last couple of weeks, I have focused on learning the basics of the two papers. Find my summaries in the following links:
- [docs/how-dreamerv3-works.md](docs/how-dreamerv3-works.md)
- [docs/how-r2-dreamer-works.md](docs/how-r2-dreamer-works.md)

![Conceptual prior and posterior distributions over possible room layouts](assets/images/prior-posterior-example.png)


---

## References

Some of the material that got me started, in case it is useful for someone else:

### Videos

- [Yann LeCun's $1B Bet Against LLMs, Part 1](https://www.youtube.com/watch?v=kYkIdXwW2AE&t=1738s) and [Part 2](https://www.youtube.com/watch?v=v_jDvpEGTIg&t=150s), by Welch Labs. On why LeCun thinks the next step for AI is World Models, not bigger LLMs.
- [Robotics' End Game: Nvidia's Jim Fan](https://www.youtube.com/watch?v=3Y8aq_ofEVs), by Sequoia Capital. On where robotics is going and the role of simulation and foundation models.

### Blogposts

- [World Models](https://worldmodels.github.io/), by Ha and Schmidhuber (2018). The interactive article that made the term popular. An agent learns to drive and to play Doom inside its own dream.
- [World Models](https://rohitbandaru.github.io/blog/World-Models/), by Rohit Bandaru. A long technical tour of World Model architectures, including the RSSM used by Dreamer, and newer ones like JEPA, Genie and Cosmos.
- [Introducing GAIA-1](https://wayve.ai/thinking/introducing-gaia1/), by Wayve. A generative World Model for self-driving, and a good example of these ideas applied in industry.

### Papers

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), by Hafner et al. This is the DreamerV3 paper and the conceptual foundation of the study. The author's [project page](https://danijar.com/project/dreamerv3/) has the uncut videos of the Minecraft diamond runs.
- [R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation](https://openreview.net/forum?id=Je2QqXrcQq), by Morihira et al. (ICLR 2026). This separate paper introduces R2-Dreamer and the `r2dreamer` PyTorch codebase used for the experiments.
