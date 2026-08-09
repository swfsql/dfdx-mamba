# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust library implementing a single [Mamba-1](https://arxiv.org/abs/2312.00752)
block on top of the [dfdx](https://github.com/coreylowman/dfdx) tensor framework
(via the [swfsql fork](https://github.com/swfsql/dfdx)). The goal is a **minimal,
readable port** of `mamba-minimal` — plain dfdx tensor ops, **no custom kernels**,
no parallel scan.

Scope is deliberately small: **one block**, no networks/layer stacks/training loop.
A sibling, far more extensive library on Burn lives at `../burn-mamba/`
(Mamba-1/2/3, chunkwise SSD, composition types); consult it for the math, but note
the two share no code and dfdx's type-level shapes make the style very different.

## Build & Test Commands

```bash
cargo +stable check                        # dfdx on stable; the block itself is NOT compiled
cargo +nightly check --features "nightly"  # the real check — compiles src/layer.rs
cargo +nightly check --features "nightly,safetensors"
cargo +nightly doc --no-deps --features "nightly"
```

- **Both checks must pass.** `rust-toolchain.toml` pins nightly, so `+stable` is
  needed explicitly for the first one.
- **`nightly` gates the entire library.** `src/layer.rs` needs
  `generic_const_exprs`; without the feature `lib.rs` exports nothing, so a bare
  `cargo check` proves almost nothing.
- `safetensors` adds `Save/LoadSafeTensors` derives — the intended way to get
  weights in (see [Key Design Decisions](#key-design-decisions)).
- **There is no test suite**; `cargo test` runs zero tests. Verification today is
  type-checking plus the downstream example (`../mamba-minimal-dfdx-example`).
- The `dfdx` dependency is pinned to a **git rev of the fork**, checked out locally
  at `../dfdx` (branch `this-main`). To bump: `git -C ../dfdx log --oneline`, put the
  rev in `Cargo.toml`. Breakage after a bump is expected to be fixed here, not there.

## Documentation Maintenance (CLAUDE.md)

- Keep this file **as minimal as possible while still viable**. Prefer pointing at
  the source (`src/layer.rs` carries the per-step comments and paper citations) over
  duplicating it here. When a source file changes, update its one entry — don't grow
  this file.
- **Never use it as a changelog.** It describes the code as it *is now*; no
  individual changes, migrations, "used to be / now", "verified by", dates, or PR
  history. If you catch changelog-style prose, delete it.
- Always be **extremely succinct** when adding content.
- **Commit messages**: the user may ask for a commit message for the session.
  **Just write the message as text** (a title line + a short body) for the user to
  copy — do NOT run `git commit` or any git command that creates the commit.
  End the message with the `Co-Authored-By:` trailer.

## File Map

```text
src/
├─ lib.rs     crate root: re-exports; every item is `#[cfg(feature = "nightly")]`
└─ layer.rs   the whole library (~1100 lines), three scopes:
   ├─ (root)     MambaBlockConfig (Dim-generic) / MambaBlockConstConfig (const-generic
   │             alias, applies the defaults) / MambaBlock + its BuildOnDevice impl
   ├─ stateless  Module<Tensor<(Batch, Sequence, DModel)>> — whole-sequence forward;
   │             ss() computes Δ/B/C, selective_scan() is the sequential time loop
   └─ stateful   Module<(Tensor<(Batch, DModel)>, MambaStateCache)> — one-token step;
                 MambaStateCache(Config), ss_step(), selective_scan_step()
```

## Architecture

### Config → Built module

dfdx's two-stage pattern: `MambaBlockConfig` (shape-only, `Dim` type params) →
`try_build_on_device` → `MambaBlock` (holds `Tensor`s, `E: Dtype`, `D: Device<E>`).
`MambaBlockConstConfig<const D_MODEL, …>` is a type alias that fills in the defaults
(`DState=16`, `DtRank=(DModel+15)/16`, `DConv=4`, `DInner=DModel*2`).

Sub-modules are dfdx `MatMul` (bias-less linear), `Conv1D`, `Bias1D`, `Linear`;
`a_log` and `d` are raw `#[param]` tensors.

### Dual execution modes (two `Module` impls on the same struct)

- **`stateless`** — `(Batch, Sequence, DModel) → (Batch, Sequence, DModel)`:
  `in_proj` → split `(xs, res)` → depthwise `conv1d` (padded `DConv-1`, then the
  tail is **split off** to restore `Sequence`) → `Bias1D` → SiLU → `ss` → `* silu(res)`
  → `out_proj`.
  `ss` projects `x_proj` into `[Δ | B | C]`, pushes Δ through `dt_proj` + softplus,
  then `selective_scan` discretises (ZOH `Ā=exp(Δ·A)`, Euler `B̄=Δ·B`) and runs a
  **sequential Rust `for` loop** over the sequence: `unstack` into per-step tensors,
  `xs = xs·Āₜ + B̄ₜuₜ`, `yₜ = xs·Cₜ`, then `stack` back and add `D·u`.
- **`stateful`** — `((Batch, DModel), cache) → ((Batch, DModel), cache)`: the
  recurrent decode step. The conv is **not** run as a `Conv1D`: the cache's conv
  window is rolled (split off column 0, concat the new column) and the kernel is
  applied as a broadcast multiply + `sum` over the `DConv` axis.

The two are separate hand-written code paths; nothing asserts they agree (no tests).

### Cache

`MambaStateCache { conv_state: (Batch, DInner, DConv), ssm_state: (Batch, DInner,
DState) }`, built from `MambaStateCacheConfig` as zeros. Only `stateful` uses it.

## Key Design Decisions

- **Type-level shapes.** Every dimension is a `Dim` type param, so dimension
  arithmetic is expressed as trait bounds (`DInner: Mul<C2>`, `DConv: Sub<C1>`,
  `DtRank: Add<<DState as Mul<C2>>::Output>`, …). These `where` blocks are repeated
  on the config, the struct, `BuildOnDevice`, and both `Module` impls — **when you
  change one, change all of them**, and expect the error messages to be long. `C1`,
  `C2`, `C4`, `C15`, `C16` are `Const<N>` aliases at the top of `layer.rs`.
- **Nightly-only** (`generic_const_exprs`), because the defaults are computed
  (`(D_MODEL+15)/16`, `D_MODEL*2`).
- **Tape-generic.** All ops go through `try_*` and thread `T: Tape<E, D>`; params are
  pulled into the tape with `.retaped::<T>()`. Autodiff therefore works, but there is
  no optimizer/training code here.
- **Weights are loaded, not initialized.** `try_build_on_device` zeroes `a_log` and
  `d` (so `A = -exp(0) = -1` — not the S4D-real init); real weights come from
  safetensors. Training from scratch would need a proper init first.
- **No parallel scan, no kernels** — the sequential loop mirrors `mamba-minimal`;
  correctness/readability over speed.
- **Open TODOs are in-source** (`out_proj`/`conv1d` bias variants, softplus
  threshold, `try_realize` unreachable arms). Don't restate them here.
- The project root is `/shared/claude/dfdx-mamba/`; siblings under `../` are
  **read-only reference** — never write outside the project root.
- When a source file is added/removed/changed, prepare an update to its entry in the
  [File Map](#file-map) (per the maintenance rules above).
  Important rule: this is reserved to the end of your workload, and if by then you
  haven't yet read this file, **do not** read it. Your context then is still big from
  the work and it is expensive to read big files then. Instead, just prepare a
  `tmp.md` file containing what would be the new [File Map](#file-map) entry, and an
  overview of the most important aspects of the created/removed/updated files, while
  being succinct. After a full context reset, manually triggered by me, we actually
  update this file.

## Notation

Shapes live in the **types**, not in names: locals are annotated with their full
`Tensor<(A, B, C), _, _, _>` type at nearly every `let`, and the codebase is
deliberately verbose about it. Paper symbols appear in comments only.

| Type param | Meaning | Paper | Default |
|------------|---------|-------|---------|
| `Batch` | batch size | `B` | — |
| `Sequence` | sequence length | `L` | — |
| `DModel` | hidden dimension | — | — |
| `DState` | latent state dim | `N` | 16 |
| `DtRank` | rank of Δ (§3.6) | — | `(DModel+15)/16` |
| `DConv` | conv kernel width | — | 4 |
| `DInner` | `DModel · expand` | `D` | `DModel·2` |
| `E` / `D` / `T` | dtype / device / tape | — | — |

Note the collision: dfdx's `D` is the *device* generic, while the paper's `D` is
`DInner`.

## Extra References

Read-only, under `../`: the **dfdx fork** this pins to (`../dfdx`, branch
`this-main`); the **downstream WASM inference example** that consumes this crate
(`../mamba-minimal-dfdx-example`); the **Burn sibling library** (`../burn-mamba`, see
its `CLAUDE.md`/`files.md`); **papers** (`../papers/`); **Python reference impls**
(`../py/`). `README.md` lists the upstream ports this was derived from
(`mamba-minimal`, candle's `mamba-minimal`, `mamba.c`, `mamba-cpu`).

## Custom Commands

- `rg`: available.
- `git`: read-only inspection (`log`/`diff`/`show`/`status`) is fine; never create
  commits, and never touch `../dfdx`'s working tree.
