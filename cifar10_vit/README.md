# CIFAR-10 Vision Transformer for the Basys 3

A second, independent accelerator architecture in this repository. The original
design targets MNIST and reaches 77.21% on hardware; this one targets CIFAR-10
(32×32 RGB, 10 classes), which is a substantially harder problem, and is built
around a different engine because the MNIST design's bottleneck turned out to be
the wrong thing.

Everything here is separate from the MNIST tree — that design still builds and
runs unchanged.

## Why a new architecture rather than a bigger old one

The MNIST accelerator's post-synthesis numbers say exactly what is wrong with it:

| Resource | Used | Available | |
|---|---|---|---|
| Slice LUTs | 18,523 | 20,800 | **89% — saturated** |
| Slice registers | 27,245 | 41,600 | 65% |
| Block RAM (BRAM18) | 6 | 100 | **6% — idle** |
| DSP48E1 | 34 | 90 | 38% |

It runs out of LUTs while the block RAM and most of the DSPs sit unused, because
its weights live in LUT-based distributed ROM and its GEMM engine is a single
sequential multiply-accumulate stepping through M×N×K cycles at 20 MHz. Scaling
that design to CIFAR-10 — 3,072 pixels instead of 784, in colour — makes the
saturated resource worse and the idle ones no less idle.

So this design moves the weights into real block RAM and the arithmetic onto a
parallel DSP array, and spends the freed LUTs on control rather than storage.

## Architecture

```
32x32x3 image
  |
  v  8x8 patches -> 4x4 grid = 16 tokens of 192 values
[patch projection 192->64]  + learned positional embedding
  |
  v
[fused QKV projection 64->192]
  |
  +-- [Q.K^T] -> [softmax] -> [.V] -> [output projection 64->64]
  |                                            |
  +--------------------- residual add ---------+
  |
  v  [LayerNorm]
  |
  +-- [FC1 64->128] -> [GELU] -> [FC2 128->64]
  |                                     |
  +------------- residual add ----------+
  |
  v  [LayerNorm] -> [global average pool] -> [classifier 64->10] -> argmax
```

Single encoder layer, single head, no CLS token (the head global-average-pools),
int8 Q1.7 throughout. 47 KB of int8 parameters.

### Why this shape

The patch projection costs `#pixels × d_model` multiply-accumulates — 3,072 × 64
here — and **that total does not depend on the patch size**. Splitting the image
into more, smaller patches does not make the stem cheaper; it only makes
attention and the FFN more expensive. Meanwhile attention scales with
`seq_len²` and the FFN with `seq_len × d_model × d_ff`, so token count is the
knob that actually moves the back half of the network.

8×8 patches give 16 tokens: enough spatial structure for attention to have
something to do, a trivial 16×16 score matrix, and a token count that divides
exactly into the PE array so no row ever idles.

### The engine

An **output-stationary 16×4 array of 64 multiply-accumulate cells**, one DSP48E1
each. Rows map to tokens, columns to a tile of four output channels. Each cycle
one activation broadcasts along its row and one weight down its column, so 20
operands drive 64 multipliers. Nothing moves between cells, so there is no
systolic skew to get right and no pipeline to drain.

Three decisions make it keep up:

- **Activations are stored feature-major.** One memory word holds one feature
  across all 16 tokens — exactly the slice the array consumes each cycle. A
  token-major layout would need 16 reads per cycle and make the array
  memory-bound. The cost is that results must be transposed on writeback, which
  the sequencer does while the next tile accumulates.
- **Weights are pre-interleaved offline** into array-consumption order, so
  weight fetch at run time is a bare address increment that never stalls.
- **Accumulators are double-banked**, so a finished tile is read out while the
  next accumulates. Measured overhead in simulation: about 2%.

The two attention products multiply activations by activations rather than by
weights. Because activations are feature-major, the weight operand is just a
four-wide slice of an activation word — attention needs a mux and a second read
port, not a separate datapath.

## Cost model

Per image, at 100 MHz:

| Stage | MACs | Array cycles | PE utilisation |
|---|---:|---:|---:|
| patch embed | 196,608 | 3,072 | 100% |
| fused QKV | 196,608 | 3,072 | 100% |
| Q·Kᵀ | 16,384 | 256 | 100% |
| softmax·V | 16,384 | 256 | 100% |
| output projection | 65,536 | 1,024 | 100% |
| FC1 | 131,072 | 2,048 | 100% |
| FC2 | 131,072 | 2,048 | 100% |
| classifier | 640 | 192 | 5% |
| **GEMM total** | **754,304** | **11,968** | **98%** |
| softmax (16 rows) | | 544 | |
| LayerNorm (×2) | | 288 | |
| GELU, residuals, pooling | | 384 | |
| **Total** | | **13,184** | |

**131.8 µs per image, about 7,600 images/second.**

The MNIST design computes ~173,000 MACs at roughly one per cycle at 20 MHz —
about 8.6 ms, or ~116 images/second. So this is roughly **65× the throughput on
a 4.4× larger workload**, which is about 290× per multiply-accumulate. The gain
is 64 parallel MACs, a 5× clock increase, and near-full array utilisation.

> The 100 MHz figure is a design target, not a timing-closure result: Vivado is
> not available in this environment (see *Status* below). The cycle counts are
> derived from the architecture and confirmed in simulation for the stages that
> are built.

## Accuracy

**37.57%** on the full CIFAR-10 test set (10,000 images).

The quantisation-aware training model and the pure-integer golden model agree
exactly on that number, which is the property that matters: it means the
training emulation and the hardware numeric contract have not drifted apart, so
the RTL — verified against the golden model — inherits a reference that is known
good. Predictions are spread across all ten classes.

For context: this is one encoder layer with 47 KB of int8 weights, and CIFAR-10
is far harder than MNIST. Published sub-1M-parameter transformers on CIFAR-10
land well above this, but they are 5–20× larger and do not have to fit a
20,800-slice Artix-7 alongside a UART and a display controller. The accuracy
knobs not yet turned are listed under *What's next*.

## Numeric contract

Everything is signed 8-bit Q1.7 — integer `v` represents `v/128`. Products
accumulate in 32 bits and requantise with an arithmetic right shift and
saturation.

Two details are pinned down deliberately, because they are the usual source of
simulation-versus-hardware mismatches:

- **Shifts round toward negative infinity**, not toward zero. Python's `>>`,
  numpy's `>>` and VHDL's `shift_right` on a signed value all floor, so the
  three implementations agree on negative operands without special cases.
  Implementing the LayerNorm output shift as a divide by `2**n` instead would be
  wrong by one on every negative element.
- **Saturation clamps, it does not wrap.** −300 becomes −128.

Two approximations replace expensive operations, and QAT trains with both in the
loop so the weights adapt to them:

- **LayerNorm has no divider and no square root.** A leading-one detector on the
  variance gives `log2(var)`; half of that is `log2(std)`, so dividing by the
  standard deviation becomes a shift. The effective divisor is rounded to a power
  of two.
- **The softmax exponential is a single 129-byte table lookup.** The reference
  clamps the magnitude of `score − row_max`, scales it into a 256-entry Q16
  table, reverses the index and shifts Q16 down to Q1.7 — but every step depends
  only on that magnitude, and just 26 of the 256 entries are reachable.
  Evaluating the reference across its whole input domain collapses the entire
  path into one table, bit-identical by construction.

## Status

Verified in GHDL simulation, bit-exact against the Python golden model:

| Module | What is proven |
|---|---|
| `pe_array.vhd` | 448 accumulators across 7 tiles: widest K (192, the overflow worst case), narrowest K (16), bias preload, mid-tile stall, back-to-back bank swap |
| `gemm_seq.vhd` + memories | A complete matmul: 1,024 outputs bit-exact vs `matmul_q`, in 1,047 cycles against an ideal of 1,024 |
| `softmax.vhd` | 7,712 values across 482 adversarial vectors (constant rows, single spikes, rows straddling the magnitude clamp, a sweep covering all 256 magnitudes) |
| `layernorm.vhd` | 32,768 values across 32 cases (zero variance, both int8 extremes, outliers, mostly-negative rows, variances either side of a power of two); all seven reachable shift values exercised |
| `elem_op.vhd` | 5,120 values across all four modes: positional add, residual add, GELU (sweeping all 256 table entries), and global average pooling |
| `uart_rx/tx.vhd` | Loopback, all ten test bytes bit-identical |

Each testbench was checked for teeth — removing the weight file or perturbing a
single expected value makes them fail rather than pass vacuously.

The software stack is complete and reproducible end to end: CIFAR-10 loading,
QAT training, int8 export, golden-model evaluation, and ROM image generation.

**Not yet built:**

- `vit_core.vhd` — the controller that chains the eight matmul stages together
  with the elementwise, softmax and LayerNorm passes between them. Every unit it
  would sequence is verified; the sequencing itself is not written.

  Working through the integration surfaced one real constraint that is not
  obvious from the block diagram, and that whoever writes this next needs to
  know. The array computes `out[m][n] = Σₖ a[m][k]·w[k][n]`, where the `a`
  operand is one feature-major word and the `w` operand must come from **one**
  word sliced down to the tile's four channels. The two attention products
  behave differently under that rule:

  | product | shape (M,K,N) | `w` operand | fits? |
  |---|---|---|---|
  | `Q·Kᵀ` | 16, 64, 16 | `K[j][d]` — `j` varies with the tile, `d = k` picks the word | yes, one word sliced |
  | `P·V` | 16, 16, 64 | `V[j][d]` — `d` varies with the tile, `j = k` picks the *byte* | **no** — spans four words |

  So `V` has to be token-major while the QKV projection that produces it writes
  feature-major. The cheap fix is not a separate transpose pass: `gemm_seq`
  already buffers a tile before writeback, so buffering four tiles (a 16×16
  byte array, 256 bytes) and writing them token-major transposes `V` for free,
  with no extra cycles. The same buffer serves the softmax, which normalises
  along the axis the score matrix is *not* stored along.

  This does mean touching a currently-verified module, which is why it was left
  rather than rushed.
- The Basys 3 top level, and Vivado synthesis. **No resource or timing numbers
  here come from synthesis** — Vivado is not installed in this environment. The
  design intent is roughly 65 of 90 DSPs (64 for the array, one for the
  LayerNorm mean-square), about 32 of 100 BRAM18 (20 for the weight ROM, 10 for
  the two activation memory copies, 2 for bias and positional embeddings), and
  LUT usage well below the MNIST design's 89% since the weights are no longer in
  LUT ROM — but those are estimates until someone runs the tools.

## Layout

```
rtl/    vit_pkg.vhd      generated: dimensions, fixed-point constants, tables
        stage_map.vhd    generated: weight ROM address map
        pe_array.vhd     the 16x4 output-stationary MAC array
        gemm_seq.vhd     drives one matmul end to end
        act_ram.vhd      feature-major activation memory
        weight_rom.vhd   pre-interleaved weight store
        elem_op.vhd      positional/residual add, GELU, average pooling
        softmax.vhd      integer softmax
        layernorm.vhd    leading-one-detector LayerNorm
        uart_rx/tx.vhd   115200 8N1
        seg7_display.vhd four-digit display driver
sw/     model_config.py  single source of truth; generates vit_pkg.vhd
        quant_ops.py     the integer primitives
        golden_model.py  bit-exact reference
        train_qat.py     quantisation-aware training
        export_weights.py ROM image generation
        eval_golden.py   test-set evaluation
        cifar_data.py    dataset loading
        gen_*_vectors.py testbench vector generation
sim/    Makefile         GHDL build; regenerates vectors, runs every testbench
        tb_*.vhd         self-checking testbenches
```

## Build and run

```bash
# Dataset (about 170 MB)
mkdir -p ../data && cd ../data \
  && curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz \
  && tar xzf cifar-10-binary.tar.gz

# Simulation: regenerates vectors from the golden model, runs every testbench
cd sim && make

# Software checks
cd sw
python3 quant_ops.py         # numeric primitive self-tests
python3 golden_model.py      # reference self-tests
python3 model_config.py      # cost model; --export regenerates vit_pkg.vhd

# Train and export
python3 train_qat.py --epochs 80        # about 9 minutes on 4 CPU cores
python3 eval_golden.py                  # integer accuracy on the test set
python3 export_weights.py               # ROM images + stage_map.vhd
```

## What's next

In rough order of value:

1. **Finish the top level** (`vit_core.vhd`, `elem_op.vhd`, the Basys 3 wrapper)
   and check a whole image end to end against `golden_model.predict(trace=True)`,
   which already returns every intermediate tensor for exactly this purpose.
2. **Run Vivado** and replace the estimates above with real utilisation and
   timing numbers.
3. **Accuracy.** 37.57% is a floor, not a ceiling — the model is small and the
   training recipe is deliberately plain (80 epochs, flip augmentation only).
   Worth trying, roughly in order of expected return: a second encoder layer
   (the cost model says about +5,400 cycles, still under 200 µs); a strided
   convolutional stem instead of a linear patch projection; more capacity in
   `d_model`; and stronger augmentation with a longer schedule.
4. **Feed the board from SPI flash** for throughput measurement. At 115,200 baud
   a 3,072-byte image takes ~267 µs to arrive but only ~132 µs to classify, so a
   UART demo is I/O-bound by about 2×; the compute rate needs a faster source to
   be visible.
