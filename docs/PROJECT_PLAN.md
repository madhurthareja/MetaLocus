# Autonomous Egocentric Hazard Agent

Meta Global AI Developer Hackathon · Track: **Indie developer** (team of 2) · Theme: leading with **Push the Frontier** (technical-depth/benchmark narrative fits better than Build for Your World's domain-problem framing, but it's a pitch choice, not a hard category — revisit if desired) · Inference box: NVIDIA DGX Spark (GB10, 128GB unified memory, ARM64/Grace-Blackwell)

**Key dates:** kickoff Oct 26 · workshops/office hours Oct 27–Nov 4 · **submission due Nov 4, 9:00 PM ET** · judging Nov 4–8 · finals/closing ceremony Nov 9. Today is Oct 28 — 2 of the hackathon's 9 total days are already gone; the build schedule in §10 reflects the ~7.5 days actually remaining, not a fresh 9-day start.

**Sanctioned toolset (official):** Meta Model API gives Muse Spark series, Muse Wonder, Muse Image, and SAM 3.1 (text-prompt route, cost-prohibitive for video — see §5). Separately, "run Meta open weights models" covers our actual SAM 3.1 Object Multiplex usage (local, gated HF checkpoint) and Muse Glimmer 30B (local). Be precise about this split in the submission writeup: Spark + Wonder via the Model API as intended; SAM 3.1 + Glimmer as local open-weights, not via the API.

**Inference partner credits:** $100 each from Baseten, Modal, Fireworks (Baseten also gives 2M TPM rate limits) — a backup/buffer if the DGX Spark stalls, not primary compute. Fireworks includes a workshop from Sinan Ozdemir worth attending.

## 1. The pitch

An edge-compute assistive navigation system: first-person video → open-world segmentation → persistent 3D spatial memory → an always-on local agent that warns a visually impaired user of physical hazards via TTS, with no query required.

**Underlying technical claim** (the evidence for "Technical Depth," not just the demo):

> Frame-level VLM reasoning over egocentric video fails systematically on spatial questions once the referent leaves the field of view or the query requires a reference frame; grounding the VLM in a persistent, segmentation-derived 3D scene graph recovers that capability — including hazard detection for objects that re-enter the user's path after leaving view.

**Prior art to cite, not compete with:** SAM 3 Agent (MLLM proposes noun phrases → SAM 3 masks → iterate, 76.0 gIoU on ReasonSeg) already does "VLM writes SAM's vocabulary" for single images with no state. Our contribution is the temporal/persistence extension, now applied to a physical-safety task: spatial working memory over a concept-segmentation backbone, driving autonomous hazard alerts instead of answering one-off queries.

## 2. Why this framing over a pure benchmark

A teammate proposed an "Autonomous Egocentric Hazard Agent" built on EgoObjects + a trained GATv2 scene graph. Two pieces of that were reconsidered and the decisions below reflect what was agreed:

- **Dataset: AEA only, not EgoObjects.** EgoObjects is Meta's own egocentric dataset and is well-aligned optically with the sponsor, but it ships frames without ground-truth depth, 6-DoF pose, or a point cloud. Hazard detection is fundamentally a distance/closing-speed problem — without AEA's free depth you'd need to reintroduce a monocular depth model, which is exactly the fragile component AEA lets us delete (see §3). AEA's cluttered kitchen/laundry-pile sequences are reframed as the hazard scenarios for this project (a chair leg, a bag on the floor, a pet, a low table), which is a reasonable proxy for "obstacles in a visually impaired user's path" without needing a literal mobility dataset.
- **No GATv2.** The original plan to train a GATv2 on Glimmer-extracted spatial triplets has no labeled training signal available in the timeline, and an undertrained GNN is a worse hazard detector than honest geometry. Time-to-collision is computed directly: `distance ÷ closing speed`, from the 3D positions already produced by the depth/pose pipeline. This is fully explainable to a judge and costs zero training time. "Scene graph" in this plan means an explicit, geometrically-built graph (nodes = tracked objects with 3D position + confidence decay, edges = computed spatial/kinematic relations), not a learned GNN.

## 3. Dataset

- **Primary, scored:** Project Aria Everyday Activities (AEA) via `projectaria_tools`. ~8 of 143 sequences, prioritizing cluttered/occluded scenes (kitchen counters, laundry piles, hallways with obstacles) as hazard proxies.
  - Gives for free: 1kHz 6-DoF trajectory, semi-dense static-scene point cloud, per-frame 3D eye gaze, aligned speech.
  - Report main numbers on the **open-loop** trajectory (real-time odometry), not closed-loop — the honest "this would run on real glasses" claim.
  - Depth: project semi-dense points into each SAM mask, take the median as object depth (monocular fallback only for masks with too few points). This same depth feeds both the scene-graph positions and the TTC hazard computation.
  - RGB is fisheye — undistort via `projectaria_tools` calibration, which also yields pinhole intrinsics K.
- **Demo-only, qualitative:** scripted phone walkthrough for the split-screen video (see §7). Labeled explicitly as qualitative, not part of the scored benchmark.
- EgoObjects: considered and rejected for the reasons in §2. Revisit only if AEA scenes prove too depth/pose-rich but too hazard-poor and extra object-category variety is worth the depth tradeoff — not expected to be needed.

## 4. Benchmark (technical-depth evidence underneath the demo)

**Question taxonomy** (5 types; hazard-relevant ones in bold):

| Type | Example | Purpose |
|---|---|---|
| In-view relational | "Is the bottle left of the kettle?" | Control — baseline should mostly pass |
| **Out-of-view recall** | "Where is the mug now?" / "is the chair still in my path?" (asked 8s after looking away) | Headline result — baseline ≈ 0; directly the hazard-reentry case |
| Reference-frame | "My left, or the laptop's left?" | SAM has no referring-expression support — graph does |
| Instance persistence | "How many distinct chairs did I pass?" | Identity tracking, not just detection |
| **Change/hazard detection** | "What moved into my path since I last looked here?" | Drives the TTS trigger logic directly |

**Ablation ladder** (A/B/C must run on the same frames as D):
- A — Spark 1.3 on current frame only (stateless captioner)
- B — Spark 1.3 on sliding window of last N frames (the "just add context" objection — must beat this)
- C — Spark + SAM boxes only, masks discarded (free from the same detector pass)
- D — Full system: masks + persistent 3D graph + geometric TTC

**Where the thesis lives or dies:** C vs. D, on clutter/occlusion scenes where box centroids land on background but masks don't. Report honestly if C ≈ D — A/B vs. D still carries the project.

## 5. Model stack

| Model | Role | Notes |
|---|---|---|
| Muse Spark 1.3 | Concept proposer, ~0.5 Hz; parses queries into (anchor, relation, target); also picks which 4-6 concepts get a live SAM 3.1 session | API, $1.25/$4.25 per M |
| SAM 3.1 Object Multiplex (native, local) | Segmentation backbone — **required by this hackathon.** One session per concept (native path doesn't support multi-concept-in-one-pass); each session multiplexes up to ~128 instances of its one concept | Local on DGX Spark, gated HF checkpoint. 4-6 live concepts = 4-6 parallel sessions, each rerunning the Perception Encoder — benchmark actual per-session cost on GB10 on day 1 (§8), since this scales differently than the single shared-encoder pass we'd use with plain SAM 3 |
| Muse Glimmer 30B | **Always-on local agent.** Continuously consumes the geometric scene graph + TTC values; triggers a hazard alert only when a threshold is crossed (not query-driven) | Local, Apache 2.0. Should fit comfortably in 128GB unified memory |
| Muse Wonder (`muse-wonder-1.0`) | Query-answering layer over the graph, for the benchmark eval and any ad-hoc demo queries | API, keep call volume low ($20/M output) |
| TTS (Muse Voice or local) | Converts Glimmer's hazard alert into spoken warning | Needed for the actual "visually impaired" use case — confirm which TTS is used (Muse Voice Transcribe is STT only; need a TTS counterpart or a local/open TTS fallback) |

Do **not** use: SAM 3.1 via Meta Model API for video (text-prompt-only, ~$270/hr at 30fps — financially dead; the local Object Multiplex checkpoint is the only viable video route). Do not use a GATv2/GNN (see §2).

## 6. Architecture

1. SAM 3.1 runs 4-6 parallel native sessions (one per concept slot), each ingesting AEA frames with its own text prompt, concepts chosen by Spark (~0.5Hz refresh — slots, not a fixed list).
2. Masks + undistorted frame → project AEA point cloud into each mask → median depth → 3D position in world frame via AEA 6-DoF pose (open-loop).
3. Scene graph: persistent nodes (track IDs from SAM), confidence decay once a node exits FoV, edges computed geometrically (left-of/right-of/behind/distance/closing-speed).
4. **Hazard loop (new):** Glimmer runs continuously over the graph, computing/consuming TTC per tracked object relative to the user's trajectory; crosses a threshold → emits a hazard event → TTS speaks a warning. This loop runs independent of any user query.
5. Separately, Wonder answers free-form queries against the graph for the benchmark eval (§4) and any demo Q&A.

## 7. Demo

Split-screen recording: raw AEA/phone frame on the left; SAM 3.1 masks + the live scene graph + the running audio/TTS transcript on the right. This keeps the demo deterministic (no live camera risk) while still showing the full pipeline end-to-end, including a staged hazard event and its spoken warning.

## 8. Hardware: DGX Spark (GB10)

- **Memory capacity is not the constraint.** 128GB unified memory comfortably holds 4-6 parallel SAM 3.1 sessions + Glimmer 30B + depth fallback loaded simultaneously.
- **Throughput/bandwidth might be, and hasn't been checked yet.** GB10's bandwidth profile (~273GB/s class) differs from a 4090/3090. With SAM 3.1's one-session-per-concept architecture, each live concept reruns the Perception Encoder — the real constraint for a live-feeling demo is sustained FPS across all parallel sessions, not whether they fit in memory.
- **ARM64 (Grace/aarch64) is a day-1 risk.** Standard PyPI wheels for PyTorch/Ultralytics/SAM deps are mostly x86_64. Verify NVIDIA's NGC PyTorch container path (ships Grace-Blackwell/aarch64 builds) before assuming plain `pip install` works.

**Action for day 1:** confirm `torch.cuda.is_available()` and a SAM 3.1 Object Multiplex forward pass run on GB10, and clock real FPS with 4-6 concurrent concept sessions on one AEA clip before locking the live-demo frame rate or concept count into the architecture.

## 9. Immediate actions — do before end of day (2026-10-28)

1. Request gated access to `facebook/sam3.1` (Object Multiplex checkpoint) on Hugging Face — this is the required model for this hackathon and approval isn't instant.
2. Claim Baseten/Modal/Fireworks participant credits — backup GPU if the DGX Spark environment stalls.
3. Start AEA download + `pip install projectaria_tools`; select 8 sequences prioritizing clutter/occlusion scenes usable as hazard scenarios.
4. Stand up the DGX Spark environment and smoke-test: load SAM 3.1 Object Multiplex natively, run one session on one AEA frame, confirm aarch64/CUDA compatibility, then test 4-6 concurrent sessions and measure FPS.
5. Decide the TTS component (Muse-native vs. a local/open TTS) — needed for the core hazard-alert loop, not just a nice-to-have.

## 10. Build schedule — actual remaining days (Oct 28 – Nov 4)

Two of the hackathon's nine days (Oct 26–27) are already spent on kickoff/early workshops. What's left is Oct 28 through the Nov 4, 9:00 PM ET deadline — treat this as ~7.5 working days, with the last day being a partial one before submission, not a full buffer day.

- **Oct 28 (today):** DGX Spark environment smoke test (§8/§9.4), HF gated access requested, AEA sequences selected and downloading, TTS component chosen.
- **Oct 29:** SAM 3.1 multi-session (one per concept) pipeline end-to-end on one AEA sequence → masks + track IDs per session. Fisheye undistortion + intrinsics wired up.
- **Oct 30:** Depth lifting (point-cloud projection into masks) + world-frame transform using AEA open-loop pose. First 3D positions logged.
- **Oct 31:** Scene graph (nodes, decay, geometric edges) + geometric TTC computation.
- **Nov 1:** Glimmer always-on hazard loop wired to the graph + TTC; threshold tuning; TTS wired to hazard events. Spark 1.3 concept-selection loop running.
- **Nov 2:** Wonder query-answering layer on the graph. Build the eval harness: 5-type question set authored against the 8 AEA sequences, conditions A/B/C/D wired up.
- **Nov 3:** Run full ablation (A/B/C/D), analyze C-vs-D honestly, stage and record hazard events, record the split-screen demo video.
- **Nov 4 (submission day, due 9:00 PM ET):** Finalize submission writeup (thesis, related-work framing vs. SAM 3 Agent, ablation numbers, open-loop honesty note, real-world-impact framing), fix whatever broke, submit with margin before the deadline — don't plan to use the full day as buffer.

## 11. Key risks

- **C ≈ D (boxes as good as masks):** mitigated by scene selection (clutter/occlusion), but report honestly either way.
- **DGX Spark ARM64/throughput unknowns:** resolve on day 1, not day 5.
- **HF gated-access delay:** request today; Baseten/Modal/Fireworks as fallback compute.
- **TTS integration underspecified:** the hazard-alert loop is the core product story — don't leave TTS to the last day.
- **Scope creep:** eye gaze and voice queries are explicit stretch goals — cut first if days 6-9 are tight.
- **SAM 3.1 parallel-session cost:** since the native path reruns the Perception Encoder per concept, 4-6 concurrent sessions may not hit target FPS on GB10 — have a fallback of reducing live concept count or frame resolution ready, decided from the day-1 FPS measurement, not guessed in advance.
