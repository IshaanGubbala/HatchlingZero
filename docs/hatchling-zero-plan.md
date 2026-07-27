# HATCHLING-ZERO Development Plan (extracted .docx)




    HATCHLING-ZERO: Revised Mac-Native Development Plan

    Updated July 26, 2026


    1. Current project status

    HZ-0A has progressed beyond a toy proof of concept. The current Mac-native implementation can:

    Train a hybrid recurrent/attention language model with approximately 109.9M parameters.

    Train a comparable dense transformer baseline with approximately 95.9M parameters.

    Evaluate validation loss and perplexity.

    Benchmark autoregressive decoding.

    Use gradient accumulation for larger-model optimization.

    Save reproducible configurations, checkpoints, and benchmark reports.

    The most important current comparison is:

    Model

Parameters

Training budget

Validation loss

Perplexity

Transformer baseline

95,937,984

25 optimizer steps

3.7620

43.04

Hybrid, original tuning

109,899,648

25 optimizer steps

3.5573

35.07

Hybrid, tuned

109,899,648

25 optimizer steps

3.2711

26.34

    The tuned hybrid therefore has a meaningful early-convergence advantage at this small training budget.

    However:

    Twenty-five optimizer steps are insufficient to establish final model superiority.

    The current transformer decodes approximately five times faster.

    The larger hybrid has not yet surpassed the best 36M hybrid checkpoint on validation loss.

    The recurrent implementation remains closer to a fallback backend than the intended optimized GDN-2 backbone.

    The comparison is close in scale but not exactly parameter- or compute-matched.

    The project should now move from proving that the architecture runs to determining whether it is genuinely better under controlled conditions.






    2. Updated definition of HZ-0A

    HZ-0A should no longer be considered complete merely when a roughly 120M hybrid trains successfully.

    HZ-0A is complete only when all four requirements are met:


    Architecture requirement

    The model uses a genuine Gated DeltaNet-2 recurrent update with:

    Channel-wise decay

    Independent channel-wise erase gates

    Independent channel-wise write gates

    Periodic exact or sliding-window attention

    Dense SwiGLU feed-forward blocks

    GDN-2 specifically separates erase and write behavior, while earlier Gated DeltaNet ties them to one scalar update gate. NVIDIA reports that this provides the strongest gains in interference-heavy memory and retrieval settings, with erase selectivity contributing much of the improvement.



    Training requirement

    The approximately 110M model must:

    Train stably for a meaningful token budget.

    Beat the matched transformer on validation loss at equal tokens.

    Be competitive when quality is measured against wall-clock training time.

    Beat or clearly explain the result of the best 36M checkpoint.

    Avoid unexplained recurrent-state or gradient instability.



    Backend requirement

    The recurrent path must use one of:

    A custom MLX/Metal forward and backward kernel.

    A chunk-parallel MLX implementation that is reasonably close to the intended hardware-efficient algorithm.

    The slow token-loop fallback is acceptable for mathematical validation, but it cannot be considered the final HZ-0A backend.



    Evaluation requirement

    The model must demonstrate an advantage on at least one task that directly exercises recurrent memory:

    Key-value overwrite

    Multi-key retrieval under interference

    Recall versus distance

    Streaming state continuity

    Long-context retrieval

    Repeated correction of prior associations

    Ordinary short-context perplexity alone is not enough to validate GDN-2.






    3. Revised project structure

    Development should proceed through three parallel tracks.

    TRACK A — Scientific comparison
Does the architecture improve model quality?

TRACK B — Mac backend engineering
Can the recurrent architecture run efficiently on Apple Silicon?

TRACK C — Data and evaluation
Are the training and benchmark signals realistic and trustworthy?

    Model scaling should pause whenever one of these tracks becomes the dominant bottleneck.





    4. Phase 0: Lock down the current baseline

    Before changing the architecture, preserve the current result as a reproducible experimental baseline.


    Required actions

    Create immutable experiment definitions for:

    HZ-36M-best
HZ-110M-untuned
HZ-110M-tuned
Transformer-96M

    Each experiment record should contain:

    Git commit

    Complete YAML configuration

    Dataset manifest and hashes

    Tokenizer files and hash

    Random seed

    Parameter count

    Number of layers

    Hidden and intermediate dimensions

    Attention/recurrent layer schedule

    Sequence length

    Microbatch size

    Gradient-accumulation count

    Effective tokens per optimizer update

    Learning-rate schedule

    Weight decay

    Gradient clipping

    Precision

    MLX/PyTorch/macOS versions

    Mac hardware and unified memory

    Checkpoint path

    Validation split definition



    Add deterministic smoke tests

    Every commit affecting the model should run:

    1. Forward output shape test
2. One-step backward test
3. Finite-gradient test
4. Checkpoint save/load equivalence
5. Streaming versus full-sequence equivalence
6. Tiny-batch overfit test
7. Fixed-seed loss regression

    This prevents backend optimization from silently altering the model.






    5. Phase 1: Run a fair hybrid-versus-transformer comparison

    The next comparison must not be based only on equal optimizer steps.

    Gradient accumulation means one optimizer step can represent different amounts of computation and different numbers of tokens. Report all comparisons using at least:

    Tokens processed

    Training FLOPs or an approximate compute proxy

    Wall-clock time

    Optimizer updates


    Models

    Train:

    A. Transformer baseline: approximately 96M
B. Current hybrid: approximately 110M
C. Parameter-matched transformer: approximately 110M
D. Optional compute-matched transformer

    Model C is necessary because the current hybrid has roughly 14M more parameters.

    Model D should adjust width or depth so its estimated active FLOPs per token are close to the hybrid.



    Checkpoint schedule

    Evaluate at:

    25
50
100
150
300
500
1,000 optimizer steps

    Also record checkpoints at fixed token counts, such as:

    1M
5M
10M
25M
50M
100M tokens

    The token-based checkpoints should be treated as the primary scientific comparison.



    Required plots

    Produce:

    Validation loss versus tokens

    Validation loss versus wall-clock time

    Validation loss versus estimated FLOPs

    Validation loss versus parameters multiplied by tokens

    Training throughput versus sequence length

    Decode throughput versus generated length

    Peak unified memory versus sequence length

    Recurrent-state norm throughout training

    Gradient norm throughout training

    The current evidence suggests the larger model may simply be severely undertrained. Compute-optimal scaling work consistently shows that increasing parameter count without proportionally increasing training tokens can produce a larger but worse-trained model.



    Decision gate

    Continue with the hybrid architecture if it satisfies at least two of:

    Lower validation loss at equal tokens

    Lower validation loss at equal estimated compute

    Better memory-task performance

    Better scaling with context length

    Comparable quality per wall-clock hour after backend improvements

    Do not claim architectural superiority from the current 25-step result alone.






    6. Phase 2: Determine why the 110M model trails the 36M checkpoint

    Run controlled ablations rather than changing several variables together.


    Experiment 2.1: Equal tokens per parameter

    Train the 36M and 110M models at approximately equal tokens per parameter.

    For example:

    36M model:
X training tokens

110M model:
approximately 3.06 × X training tokens

    This will help determine whether the 110M result is caused primarily by undertraining.



    Experiment 2.2: Learning-rate sweep

    Test the 110M hybrid with:

    1.0e-4
1.5e-4
2.0e-4
3.0e-4
4.0e-4

    Keep all other settings constant.

    The successful change from the untuned model to 2e-4 plus gradient accumulation shows that optimization is currently a major variable.

    Record:

    Initial loss reduction

    Loss after a fixed number of tokens

    Maximum gradient norm

    State norm

    NaN or overflow count

    Validation loss after cooldown



    Experiment 2.3: Effective batch size

    Test gradient accumulation:

    1
2
4
8

    Compare at equal tokens rather than equal updates.

    Use FP32 gradient accumulation even when model computation uses FP16 or BF16 where supported. Mixed-precision language-model training commonly keeps numerically sensitive accumulation and optimizer operations at higher precision for stability.



    Experiment 2.4: Depth versus width

    At approximately 110M parameters, compare:

    deeper and narrower
shallower and wider

    Recurrent state matrices can become disproportionately expensive as head dimensions increase. A deeper, narrower model may give better Mac utilization and a smaller recurrent state per layer.



    Experiment 2.5: Recurrent state dimensions

    Sweep:

    key dimension: 32, 64, 96
value dimension: 32, 64, 96

    Do not automatically inherit the official 1.3B GDN-2 choice of 128-dimensional key and value heads. NVIDIA’s published configuration was designed for an H100-scale 1.3B model trained on 100B tokens, not a 110M Mac experiment.



    Experiment 2.6: Attention frequency

    Compare:

    1 attention layer per 2 recurrent layers
1 per 3
1 per 4
1 per 6
recurrent-only control

    The current default of three recurrent layers followed by one attention layer is sensible, but should be verified at this scale.






    7. Phase 3: Build the genuine GDN-2 reference

    Do not optimize GDN-2 before establishing a trusted reference.


    Reference implementations

    Create:

    gdn2_numpy_reference.py
gdn2_mlx_reference.py
gdn2_streaming_reference.py

    The implementation must preserve:

    Channel-wise decay

    Key-axis erase selectivity

    Value-axis write selectivity

    State input/output

    Masked tokens

    Full-sequence and streaming execution



    Required mathematical tests


    Forward equivalence

    Compare NumPy FP64 and MLX FP32 on small tensors.



    Streaming equivalence

    Verify that:

    full sequence
≈
multiple chunks carrying recurrent state
≈
one token at a time



    Gradient checking

    Use finite differences for:

    Query

    Key

    Value

    Decay logits

    Erase logits

    Write logits

    Initial state



    Special-case recovery

    Verify that GDN-2 reduces toward:

    KDA-style behavior when erase and write collapse to a common scalar

    Original Gated DeltaNet when decay is also scalar

    The official GDN-2 formulation is explicitly designed as a strict generalization of these earlier update rules.







    8. Phase 4: Replace the fallback backend

    This is now the highest-value engineering task.

    The transformer’s roughly fivefold decode advantage indicates that model quality work alone cannot make HZ-0A successful. The recurrent path must become faster.


    Backend path A: Adapt the trainable MLX GatedDeltaNet VJP branch

    Use the existing trainable-GatedDeltaNet work as a structural reference for:

    Custom MLX VJP registration

    Metal forward kernel integration

    Chunked backward recurrence

    State-history management

    Deterministic reductions

    Gradient validation

    The current upstream MLX-LM GatedDeltaNet path has had practical training and memory issues on Apple hardware, including first-backward-pass failures in some Qwen3.5 configurations. This reinforces the need for an HZ-specific tested backend rather than assuming standard model support is sufficient.



    Backend path B: Port GDN-2 directly to Metal

    Implement in this order:

    1. GDN-2 single-token forward
2. GDN-2 full sequential forward
3. Chunked forward
4. Custom backward/VJP
5. Chunk-parallel forward
6. Chunk-parallel backward

    Do not begin with the full chunkwise WY implementation. First obtain a correct Metal recurrence matching the MLX reference.



    Backend benchmarks

    Test operation-level shapes representative of HZ, not only Qwen-sized layers.

    Measure:

    Forward latency

    Forward-plus-backward latency

    Kernel launches per token

    Memory allocations

    State reads and writes

    Effective memory bandwidth

    Peak Metal memory

    Compilation overhead

    Performance at sequence lengths 128, 256, 512, 1,024 and 2,048



    Decode investigation

    Profile decode separately from training.

    Break token latency into:

    embedding
QKV/gate projections
causal convolution
recurrent state update
attention layers
MLP
normalization
LM head
sampling
Python/runtime overhead

    The current fivefold gap may not be entirely recurrence arithmetic. MLX-LM has previously had architecture-specific GatedDeltaNet slow paths and cache-related performance issues, so input handling, cache updates and graph specialization must also be measured.



    Backend success criteria

    Before scaling beyond approximately 110M:

    Decode slowdown should fall from approximately 5× to below 2× at short context.

    The hybrid should approach or exceed transformer decode speed as context grows, if the fixed recurrent state is delivering its intended advantage.

    Training throughput should improve by at least 2× over the current fallback.

    Kernel and reference outputs must remain numerically aligned.






    9. Phase 5: Improve the training data

    The next model-quality gains may come more cheaply from data than architecture.


    Build a documented data mixture

    Use a compact, high-quality corpus with explicit proportions:

    45% filtered educational web text
20% Wikipedia and reference text
15% public-domain books
10% code
5% science and technical text
5% synthetic memory/reasoning examples

    The exact proportions should be tested, but the dataset should not be a single homogeneous source.

    Research on open pretraining corpora has shown that diverse, deduplicated multi-source mixtures can improve domain coverage and downstream performance compared with less diverse single-source corpora.



    Required preprocessing

    Exact deduplication

    Near-duplicate detection

    Boilerplate filtering

    Language identification

    Minimum and maximum document length

    Repetition filtering

    Held-out domain-balanced validation split

    Benchmark contamination checks

    Token-count reports by source

    Stable document ordering

    Dataset version hashes



    Tokenizer audit

    Compare at least:

    16K vocabulary
24K vocabulary
32K vocabulary

    For every tokenizer, record:

    Average tokens per character

    Average tokens per word

    Code compression

    Scientific text compression

    Vocabulary parameter cost

    Validation loss in bits per byte

    At 110M parameters, a 32K vocabulary with tied 768-dimensional embeddings already consumes roughly 24.6M parameters. Tokenizer and vocabulary decisions can therefore materially change how much capacity remains for the backbone.



    Data curriculum

    Use stages:

    Stage 1:
clean, shorter, easier documents

Stage 2:
broader general text and code

Stage 3:
longer contexts and more difficult material

Stage 4:
memory interference and retrieval mixtures

    Do not introduce very long sequences before the model learns basic language structure.






    10. Phase 6: Improve the optimization recipe


    Default large-model recipe

    The current tuned configuration becomes the new baseline:

    learning_rate: 0.0002
gradient_accumulation_steps: 4
weight_decay: 0.1
gradient_clip: 1.0



    Add warmup and reusable continuation

    Use either:

    Warmup followed by cosine decay for fixed-budget experiments

    Warmup-stable-decay for runs expected to be extended

    An experimental WSqD branch only after the baseline is stable

    A newly proposed horizon-independent WSqD schedule reports competitive results when training horizons are extended, but it is very recent and should remain an optional experiment rather than replacing the established baseline immediately.



    Tune initialization

    Add architecture-specific initialization for:

    Recurrent output projections

    Decay biases

    Erase gates

    Write gates

    Residual branches

    Attention output projections

    MLP down projections

    Initial gate behavior should favor stable retention rather than aggressive erase/write operations.

    Recommended starting behavior:

    decay: long but finite memory
erase: initially conservative
write: initially moderate
residual output: small



    Add state regularization diagnostics

    Track:

    Mean decay by layer

    Mean erase activation

    Mean write activation

    State Frobenius norm

    Effective state rank

    Fraction of saturated gates

    Update-to-state norm ratio

    State similarity across adjacent tokens

    Do not immediately add regularization losses. First determine whether pathological behavior exists.



    Optional knowledge distillation

    After the architecture is stable, train HZ-0A using logits or generated data from a capable teacher.

    Compare:

    pure next-token pretraining
teacher-logit distillation
mixed hard-label and soft-label training

    Distillation may improve a 110M model more efficiently than adding parameters, but it should be introduced only after the architecture comparison is understood.






    11. Phase 7: Build a benchmark suite appropriate for HZ


    Language modeling

    Track:

    Held-out validation loss

    Perplexity

    Bits per byte

    WikiText perplexity

    LAMBADA accuracy and perplexity



    Small-model downstream tasks

    Primary:

    HellaSwag
PIQA
ARC-Easy
ARC-Challenge
WinoGrande
SciQ

    Run zero-shot as the primary configuration and five-shot as a separate result.



    HZ-specific memory tests


    Associative recall

    A → red
B → blue
query A



    Overwrite

    A → red
later A → green
query A



    Protected unrelated memories

    A → red
B → blue
overwrite A → green
query B



    Multi-key interference

    Insert many similar keys and test retrieval after distractors.



    Recall-distance curve

    Measure retrieval at:

    32
64
128
256
512
1,024
2,048 tokens



    State reset and contamination

    Verify that unrelated sessions do not inherit recurrent memory.



    Long-context evaluation

    Use selected RULER-style tasks, particularly multi-key needle-in-a-haystack tests. GDN-2’s strongest reported improvements occur in interference-heavy retrieval, making these more informative than ordinary single-needle tests.




    Efficiency scorecard

    For every model report:

    Parameters

    Active parameters

    Training tokens/sec

    Decode tokens/sec

    Prefill tokens/sec

    Peak memory

    Model size

    Recurrent-state size

    KV-cache size

    Validation loss per hour

    Validation loss per billion estimated FLOPs

    Energy use where macOS measurements are reliable






    12. Phase 8: Establish explicit go/no-go gates


    Gate A: Current hybrid viability

    Proceed to genuine GDN-2 when:

    The tuned hybrid maintains an advantage beyond 25 steps.

    The advantage remains at equal tokens.

    Training is stable through at least several hundred updates.

    The results reproduce across at least two seeds.



    Gate B: GDN-2 viability

    Keep GDN-2 only when it beats standard GDN on at least one of:

    Validation loss

    Overwrite accuracy

    Interference resistance

    Long-distance recall

    while keeping throughput within approximately 20–30% of the optimized standard-GDN backend.



    Gate C: Scaling beyond 110M

    Scale toward 200–300M only when:

    The 110M model beats the 36M model at a fair token budget.

    The backend is no longer dominated by the fallback loop.

    Training data is sufficiently large and deduplicated.

    Validation loss continues improving predictably.

    The model fits with adequate optimizer and activation headroom.



    Gate D: HZ-0B

    Do not add the Hebbian scratchpad until HZ-0A is scientifically and operationally stable.

    Otherwise it will be impossible to distinguish:

    GDN-2 gains

    Scratchpad gains

    Optimization gains

    Extra parameter gains

    Backend regressions






    13. Revised HZ model roadmap


    HZ-0A — Dense recurrent hybrid

    GDN-2
periodic exact/sliding attention
dense SwiGLU
optimized MLX/Metal backend

    Goal: prove efficient recurrent memory and a controlled advantage over a transformer.



    HZ-0B — Dense plus Hebbian scratchpad

    Add a low-rank, bounded synaptic memory with explicit reset and persistence rules.

    Goal: improve temporary associative storage without modifying permanent model weights.



    HZ-0C — Session-local fast weights

    Add test-time adaptation inside selected projections.

    Goal: compress context or task behavior into temporary weights while preventing cross-session leakage.



    HZ-0D — Adaptive internal recurrence

    Allow selected blocks to execute a variable number of latent computation steps.

    Goal: spend more compute on difficult tokens or problems and less on easy ones.



    HZ-0E — Micro-MoE

    Replace selected dense MLPs with a small sparse expert system.

    Goal: add total capacity while keeping active computation manageable.

    HZ-0A through HZ-0D should remain dense in their feed-forward layers. MoE should not be introduced until HZ-0E.






    14. Immediate execution plan


    Next experiment 1: Continue the tuned 110M model

    Continue from the tuned configuration to:

    50
100
150
300
500 steps

    Do not resume the untuned model.



    Next experiment 2: Continue the transformer control

    Give the transformer the same:

    Number of tokens

    Data order

    Sequence length

    Validation checkpoints

    Approximate wall-clock reporting

    Add a parameter-matched approximately 110M transformer if feasible.



    Next experiment 3: Run a learning-rate mini-sweep

    Use short, fixed-token runs around:

    1.5e-4
2.0e-4
3.0e-4

    Do not expand to a large sweep until the basic trend is visible.



    Next experiment 4: Add memory diagnostics

    Implement:

    associative recall
overwrite
protected unrelated memory
recall versus distance

    Run them on:

    Transformer

    Current hybrid

    Best 36M model

    Tuned 110M model



    Next experiment 5: Profile decode

    Produce a per-component latency report and identify whether the dominant slowdown is:

    Recurrent math

    Python loop overhead

    Metal graph launches

    State copies

    Convolution

    Cache handling

    LM head

    Sampling



    Next experiment 6: Begin GDN-2 reference implementation

    Build and test the pure NumPy and pure MLX versions while longer baseline runs continue.

    Do not connect GDN-2 to the full language model until its forward and gradient tests pass independently.






    15. Near-term success target

    The next credible HZ milestone should be:

    A reproducible approximately 110M dense hybrid trained entirely on Apple Silicon that beats a parameter-matched transformer at equal training tokens, demonstrates stronger memory overwrite or interference performance, and runs through an optimized differentiable Metal recurrent kernel.

    That result would justify calling HZ-0A an actual architecture contribution rather than merely a functioning hybrid experiment.

    The current tuned result is encouraging because it shows that the hybrid’s quality signal survives near-plan scale and responds strongly to better optimization. The project’s main risks are now identifiable:

    Severe undertraining

    Recurrent backend inefficiency

    Insufficiently controlled baselines

    Data quality and validation realism

    Adding later HZ features before the base architecture is understood

    The revised strategy is therefore:

    control the experiment
→ train long enough
→ improve the data
→ validate memory behavior
→ implement genuine GDN-2
→ optimize the Metal backend
→ scale only after the evidence supports it














