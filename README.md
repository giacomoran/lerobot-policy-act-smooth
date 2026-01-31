# ACTSmooth

## Delay conditioning (inference semantics)

ACTSmooth supports delay-conditioned inference where the action prefix comes from the _previous_ chunk. In real systems, inference and execution latency mean some actions are already committed but not yet executed when a new observation arrives. The prefix therefore represents guaranteed future actions, and the model predicts a continuation chunk that is consistent with that queued prefix. This yields smoother control without waiting for the full prior chunk to finish.

## References

- [Zhao, 2023 - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Tang, 2025 - VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference](https://arxiv.org/abs/2512.01031)
- [Black, 2025 - Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
