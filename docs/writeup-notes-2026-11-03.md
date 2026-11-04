# Writeup material — Nov 3 fixes (for the Nov 4 submission)

Raw material for the submission writeup and video. Every number is measured on the real SANPO session `-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG` (539 frames, 36 s, EdgeTAM). The method for each one is in `docs/CLAUDE.md`.

## Headline: what the fixes changed

| | Before | After |
|---|---|---|
| Reported side matches the side of the image the object is on | 55% (a coin flip) | 100% |
| Frames with a critical (TTC < 2 s) alert | 342 / 539 | 0 / 539 |
| "Velocity" of static objects (poles, trees, hydrants) | median 13 m/s | 0 |
| Spoken cues over the walk | 25 | 12 (11 ambient mentions, 1 genuine warning: a pedestrian walking toward the stopped wearer) |

The one-line story: **the system was confidently saying the wrong side and crying wolf on almost every frame, and nothing in the per-frame output looked wrong.** The left/right bug only showed up when spoken sides were compared against the video. The false alarms only showed up when we asked why a parked truck was doing 112 m/s.

## Lead with the attention argument

Blind pedestrians aren't short of information: a cane and ambient sound already report a lot. What's scarce is hearing, which is already doing navigation work, and every word the system speaks competes with it. That one paragraph justifies the whole alert policy. Put it early.

## Three kinds of output (name all three)

- **Push:** things that will hurt you. Unrequested, rare, short ("van, 1 o'clock, 1s"). This is what the hazard pipeline is for.
- **Ambient orientation:** landmarks nearby ahead. Sparse, once per object, never repeated ("There's a pole just ahead at 1 o'clock"). Ten of the twelve cues on the real walk are this kind, and that's honest if it's named as its own category.
- **Pull:** rich orientation on demand ("what's around me?"). Deliberately not built: we scoped to the half that can't wait to be asked for.

The clock face is common orientation-and-mobility and sighted-guide language. Say "common practice", not "validated".

## Lead time is budgeted against speech

Speaking eats the margin it announces. "Van, one o'clock, one second" is ~1.6 s of synthesized speech; the friendly narrated sentence is 2.2–2.6 s. So anything on a collision course always gets the short form, and narration is kept for ambient mentions. The critical threshold stays at 2 s. Raising it to 4 s would make every pole 5 m ahead at walking pace "critical" and rebuild the crying wolf. Moving things already get the 5 s "approaching" tier. The real fix is tones, which land in milliseconds: `render_cue` (stereo-panned, TTC-rate pulses) exists and is tested, but nothing in the pipeline plays it yet. Say that plainly as the next step.

## The demo beat: real motion that isn't danger

At ~18 s on the real walk, a van genuinely drives across in front of the stopped wearer (~2.4 m/s, measured). The system mentions it calmly instead of alarming, because it will pass ~4 m clear. That moment shows the judgment the system was built for: telling real motion apart from real danger. Don't manufacture a crisis on the real data, because there isn't a true critical in this session. The closest real moment is at ~18 s: a pedestrian walks toward the stopped wearer at ~1.4 m/s and gets a short "approaching" warning. Among the synthetic sessions, only `synthetic_cluttered_occlusion` reaches critical (walking straight into a chair). `synthetic_crossing_hazard`'s 3 s clip bottoms out at a ~3.6 s time-to-collision, so it shows "approaching", not critical. Any synthetic beat should be labelled as such and used as a supplement, not the headline. The head-turn beat is already synthetic, and two synthetic beats start to look like the real data doesn't support the claim.

## Describe, don't instruct

Speech names the object, its clock position and, if it's closing, its seconds. It never says "step left" or "stop": the wearer knows their route, and the system only knows where things are. The HUD follows suit: "VAN · RIGHT", not "STOP". Its idle state reads "NO ALERTS", never "PATH CLEAR". The system can't see curbs, holes or anything it wasn't asked to track, and false reassurance is the one error that can actually hurt someone.

## "We know what's safe to remember"

Persistence is the thesis, but remembering is only safe for things that stay put. Concepts are sorted three ways:

- **Static** (pole, tree, hydrant, bin, sign, bench…): remembered for 20 s at their last position, which is still correct after they leave view. Velocity is pinned to zero.
- **Movable** (person, car, bike, dog…): fade within ~1 s of leaving view. They get no alert once unseen for 0.75 s, because a remembered car is a guess that gets worse every frame. Their velocity comes from a line fit over the current track, restarted whenever the segmenter swaps one car for another under the same id.
- **Unknown:** the old behaviour.

## Limits to state plainly

- Confidence is one scalar per object. What a real system needs is a position-uncertainty ellipse: depth error grows with range, and at 25 m the per-frame position noise is about ±2 m.
- Memory assumes unobserved space is unchanged. That's true for buildings and false for traffic, which is exactly why movers are forgotten fast.
- The closest-approach gate (0.75 m) sits close to the measured 1.0 m miss distance of the poles we walk past. Heading error (median 14.7° against the chest camera) moves the estimate by about 0.5 m at 2 m range, so a borderline object can still flip.
- EdgeTAM swaps identities between similar objects (one "truck" id covered 19 trucks in 36 s). We contain the damage; we don't fix the tracker.
- **"0 false critical alerts" was only ever true for the one walk the fixes were tuned against.** Running the same pipeline on two more real SANPO walks it had never seen found two more real false alarms, by two different mechanisms — this is exactly why we tested on data we hadn't tuned on, and it's a stronger, more honest story than claiming perfection:
  - **Fixed:** Spark proposed "suitcase" for one walk. It matched neither the static nor movable word list, so it kept zero velocity protection — the same "trucks at 112 m/s" bug, just via an unanticipated concept. Fix: any concept outside the hardcoded vocabulary is now classified once by asking Spark itself (static or movable, and a safe top speed) rather than running unprotected; a failed classification call defaults to cautious rather than unprotected. Confirmed removed on the real walk (critical frames 23 → 19, suitcase gone from the flagged set, nothing else changed).
  - **Open, not fixed:** a parked truck's lifted position drifted toward the wearer slowly enough (about 0.07 m/frame) to pass both the sudden-jump filter and the velocity smoothing, producing a sustained ~1.2 s false critical alert even though the wearer had stopped moving. Large, close objects amplify small depth error into larger absolute position error, and no speed ceiling generous enough to permit real traffic is tight enough to catch this. Named here rather than patched under deadline pressure.

## Head-turn invariance

Clock positions come from the walking direction, not the head, so turning your head doesn't move "the pole at 2 o'clock". Show it on `synthetic_head_turn_away` / `synthetic_head_turn_return`, which have ground truth. Say plainly that none of the cached SANPO walks contains a clean head-turn-while-walking: 0 episodes over 35° in sessions 1–2.

## Vision slide: familiar routes

A persistent world model makes "what's different today?" possible on a route you walk every day: the bin that's out, the scaffolding that wasn't there. A stateless detector can't say that, because it has no yesterday.

## Next steps

- Walkable-surface relevance, as a filter on what gets mentioned and never as a "path is clear" claim.
- Automatic orientation when the wearer stops: one short summary of what's around.
- Three output registers: spatial tone, short directive, narrated sentence.
- Scope against the cane: prioritise what a cane misses (head-height obstacles, fast movers).
- Alert lead times validated with BLV users rather than set by us.
