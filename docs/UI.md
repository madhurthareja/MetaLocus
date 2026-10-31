# LOCUS — Complete UI/UX Design and Implementation Brief

## 1. Product Overview

Design and implement a premium, production-quality web dashboard for **Locus**, a persistent spatial-awareness and hazard-warning system for outdoor navigation by blind and low-vision users.

Locus is designed for Meta Ray-Ban Display smart glasses.

### Core thesis

Traditional frame-level AI captioning loses spatial context when an object leaves the camera's field of view. Locus solves this by maintaining a persistent, world-coordinate 3D memory graph of detected objects.

This allows the system to answer questions such as:

- Is something approaching?
- Where was the object last observed?
- Is it still moving?
- Is it outside the camera frame?
- Does its estimated trajectory create a collision risk?

The system uses:

1. Real camera frames.
2. Muse Spark through Meta Model API for autonomous object selection.
3. SAM 3.1 Object Multiplex for segmentation.
4. Median-depth mask lifting into 3D.
5. Persistent world-frame memory graph.
6. Temporal decay, eviction, spatial re-association and duplicate-track merging.
7. Geometric time-to-collision estimation.
8. SeamlessM4T v2 for short spoken directives and non-verbal spatial tones.

This is both a functioning assistive technology system and a research platform.

The dashboard must communicate both identities.

**The primary design objective: Make persistent spatial understanding visually tangible.**

A reviewer should immediately understand that an object can leave the camera feed, remain in world memory, and subsequently trigger a meaningful hazard warning.

---

## 2. Design Philosophy

Create a visual identity inspired by Meta's modern product ecosystem, particularly its emphasis on:

- Human-centred technology.
- Simplicity and clarity.
- Generous whitespace.
- Clean typography.
- Restrained blue accents.
- Rounded geometry.
- Fluid spatial visualizations.
- Accessible information presentation.
- High-quality, minimal interfaces.

Do not reproduce Meta's website or copy its logo. Create an original Locus identity using similar design principles.

### Desired aesthetic

Think:

- Meta product ecosystem.
- Premium scientific visualization.
- Modern autonomous systems research.
- Apple-level attention to interface detail.
- Clean research instrumentation.
- Spatial computing.

Avoid:

- Generic AI dashboards.
- Purple gradients.
- Cybersecurity control-room aesthetics.
- Excessive dark mode.
- Glowing borders everywhere.
- Unnecessary glassmorphism.
- Floating AI orbs.
- Decorative animations.
- Excessive rounded cards.
- Overloaded data visualizations.

The interface should feel like a real product that could be part of Meta's wearable technology ecosystem.

---

## 3. Visual Design System

### Colour palette

Use a predominantly light interface.

Primary colours:

- Meta-inspired blue: #0866FF
- Deep ink: #1C2B33
- Primary background: #FFFFFF
- Secondary background: #F0F2F5
- Subtle surface: #F7F8FA
- Primary text: #1C2B33
- Secondary text: #667781
- Border: #E4E8EC

Semantic colours:

- Safe / operational: #16865D
- Potential hazard: #E69B25
- Critical warning: #D94848
- Informational: #0866FF

Use semantic colours only where they communicate actual system states.

### Spatial visualization palette

The 3D world visualization should use a dedicated dark canvas.

- Background: #101820
- Point cloud: muted cyan and blue
- Currently visible objects: blue
- Remembered objects: desaturated cyan
- Predicted trajectories: amber
- Active hazards: orange
- Critical hazards: red

The dark visualization should contrast elegantly with the surrounding light interface.

### Typography

Use:
- Inter or Geist for general interface typography.
- Geist Mono or JetBrains Mono for numerical telemetry, timestamps, track IDs and coordinates.

Typography should have a clear hierarchy.

Avoid oversized headings everywhere. Use large typography only for important status messages and primary page titles.

### Geometry

- Consistent 8px spacing system.
- Moderate corner radii, generally 10–16px.
- Thin neutral borders.
- Subtle shadows.
- Consistent iconography.
- Generous whitespace.
- Minimal visual decoration.

Use Lucide icons consistently.

---

## 4. Application Architecture

Build a responsive desktop-first web application.

Primary navigation should be a fixed left sidebar.

### Application header

The top header should contain:

- Locus logo and wordmark.
- Current connected hardware: Meta Ray-Ban Display.
- Hardware connection status.
- Dataset selector.
- Session status.
- Session elapsed time.
- User/profile menu.

The header should remain compact.

### Left navigation

Five primary sections:

1. Overview
2. World Memory
3. Hazard Timeline
4. Experiment Lab
5. System

Use simple line icons and short labels.

The active navigation state should use a subtle blue background with blue text.

At the bottom of the sidebar, display a compact hardware connection card containing:

- Ray-Ban Display connection.
- Camera stream status.
- Display resolution.
- Audio output status.

Also show the currently selected dataset.

---

## 5. Page One: Overview Dashboard

This is the primary landing page and should receive the most design attention.

### Page heading

Title:

"Your surroundings, remembered."

Subtitle:

"Persistent 3D awareness beyond the camera frame."

Display three compact summary metrics:

**Tracked Objects**
- Current number of persistent memory nodes.
- Number currently visible.
- Number retained outside the camera frame.

**Active Hazards**
- Number of currently active hazard events.
- Severity indicator.

**Navigation Status**
- Current system status.
- Avoid claiming that a route is safe simply because no hazard was detected.

### Main dashboard layout

Use a two-column layout.

Left column: approximately 58–62% width.

Right column: approximately 38–42% width.

The left column should contain:

1. Interactive 3D World View.
2. Event Timeline.
3. Perception Pipeline.

The right column should contain:

1. Live Camera Feed.
2. Active Hazard Panel.
3. Ray-Ban Display HUD Preview.

### A. Interactive 3D World View

This is the signature feature of Locus.

It should be the largest visual element on the dashboard.

Build a spatial visualization showing:

- A reconstructed outdoor environment.
- A ground plane.
- 3D object positions.
- Object bounding boxes.
- Object track IDs.
- Directional motion vectors.
- Predicted trajectories.
- The camera position and orientation.
- The camera field of view.
- Previously observed objects that are now outside the camera frame.

Provide three view controls:

- 3D Perspective.
- Top View.
- Side View.

Add a compact legend:

- Currently visible.
- In memory / out of view.
- Estimated position.
- Predicted trajectory.
- Potential hazard.

### Critical interaction: Persistent memory

Demonstrate the fundamental Locus contribution.

An object must not disappear from the 3D visualization merely because it leaves the camera frame.

Its visual representation should transition:

1. Solid bounding box when observed.
2. Desaturated or ghosted representation when outside the camera frame.
3. Gradual visual fading as its memory becomes stale.
4. Distinct estimated trajectory when motion is extrapolated.

Show observation age and track confidence on selection.

Use different visual treatments for observations, retained memory and predictions.

Never represent an estimated position as an actual current observation.

Clicking an object should open a contextual inspector showing:

- Track ID.
- Object category.
- Current visibility state.
- Last observed timestamp.
- Last known world coordinates.
- Estimated velocity.
- Direction of movement.
- Memory age.
- Confidence.
- Association history.
- Related hazard events.

The visualization should support mouse rotation, zoom and object selection.

Do not make it a decorative static illustration.

### B. Live Camera Feed

Display the real camera frame.

Overlay:

- SAM segmentation masks.
- Object boundaries.
- Track IDs.
- Detection labels.
- Depth information where available.

Include:

- LIVE indicator.
- Camera resolution.
- Frame timestamp.
- Current frame index.
- Expand/full-screen control.

The camera feed and 3D world view must be synchronized.

When an object is selected in either view, highlight the corresponding object in the other view.

When an object leaves the frame, its corresponding memory node should remain visible in the world view.

### C. Active Hazard Panel

This is the most important operational information panel.

When a hazard exists, display:

- Hazard severity.
- Object category.
- Direction of approach.
- Last observed state.
- Current visibility state.
- Estimated trajectory.
- Geometric time-to-collision.
- Spoken directive.
- Spatial tone direction.

Example:

APPROACHING FROM LEFT

CAUTION LEFT

Previously tracked · Currently outside camera frame.

Include a compact visual representation of the relative direction of the hazard.

Use amber for potential hazards and red for critical warnings.

The hazard panel should explain why the warning occurred, rather than simply showing an alert.

### D. Event Timeline

Implement a synchronized horizontal timeline.

Display:

- Object detection events.
- Object leaving the field of view.
- Re-association events.
- Memory updates.
- Hazard predictions.
- Warning triggers.
- Audio output events.

Allow users to:

- Play and pause.
- Scrub backwards and forwards.
- Adjust playback speed.
- Jump to specific hazard events.

The 3D world, camera feed, hazard panel and HUD preview should all update when the timeline is scrubbed.

### E. Perception Pipeline

Display the complete pipeline as a compact horizontal sequence.

Camera → Muse Spark → SAM 3.1 → Depth Lifting → World Memory → TTC → Audio.

Each stage should have:

- Name.
- Operational status.
- Processing description.
- Latency where actual telemetry is available.
- Error state if applicable.

Selecting a stage should open more detailed diagnostics.

Do not fabricate latency or operational metrics.

---

## 6. Page Two: World Memory

This page should focus entirely on the persistent 3D memory graph.

Use a large interactive visualization with a contextual inspector.

### Main visualization

Show the environment reconstructed from object detections.

Support:

- 3D perspective.
- Top-down view.
- Object filtering.
- Track selection.
- Trajectory display.
- Memory-state filtering.
- Time-based reconstruction.

### Object inspector

Display:

- Track ID.
- Object class.
- World-frame position.
- Velocity.
- Direction.
- Last observation.
- Memory confidence.
- Visibility state.
- Track age.
- Associated detections.

### Memory operations panel

Expose the actual memory mechanisms:

- Temporal decay.
- Node eviction.
- Spatial re-association.
- Duplicate-track merging.

Include an event log showing when these operations occurred.

Make it possible to inspect why two observations were associated with the same persistent object.

This page must visually establish that Locus is maintaining a world model rather than simply accumulating frame-level detections.

---

## 7. Page Three: Hazard Timeline

Create a research-grade event reconstruction interface.

The objective is to let researchers inspect individual hazard events.

Each event should have:

- Event ID.
- Timestamp.
- Object ID.
- Object category.
- Last visible frame.
- Out-of-view duration.
- Estimated motion.
- TTC calculation.
- Warning threshold.
- Generated directive.
- Audio event.

Clicking an event should open a detailed reconstruction.

Show:

1. Initial detection.
2. Segmentation.
3. Depth lifting.
4. World-frame association.
5. Object leaving the camera frame.
6. Memory retention.
7. Geometric collision prediction.
8. Warning generation.

Display a synchronized camera and 3D world replay.

The timeline should clearly distinguish actual observations from estimated states and predicted trajectories.

---

## 8. Page Four: Experiment Lab

This is the scientific evaluation area.

It must look like a professional research analysis tool rather than a generic analytics dashboard.

The main research comparison is between four experimental conditions, A/B/C/D.

Support:

- A: Stateless baseline.
- B: Intermediate ablation.
- C: Intermediate ablation.
- D: Full persistent-memory system.

Use the actual experimental definitions from the research implementation. Do not assume or invent what each intermediate ablation removes.

### A. Ablation Studio

Compare:

- Out-of-view answer rate.
- Hazard detection performance.
- False warnings.
- Missed hazards.
- Track fragmentation.
- Re-association performance.
- TTC prediction performance.
- Processing latency.

The reported central result is:

Stateless captioning: 0% out-of-view answer rate.

Persistent memory: 100% out-of-view answer rate.

Represent this result prominently, but clearly label it with its evaluation protocol and denominator.

Include confidence intervals where available.

Never manufacture statistical significance or unreported metrics.

### B. Dataset Evaluation

Provide separate evaluation sections for:

**SANPO**

Primary outdoor dataset for blind and low-vision navigation research.

Show:

- Dataset information.
- Evaluation coverage.
- Outdoor scenes.
- Relevant sensor modalities.
- Evaluation metrics.
- Experiment results.

**AEA**

Indoor dataset used as a cross-dataset generalization check.

Show:

- Dataset information.
- Sensor differences.
- Depth modality differences.
- Evaluation coverage.
- Cross-dataset results.

Make it clear that the same pipeline is evaluated across different sensor and depth modalities.

### C. Failure Analysis

Include:

- False positive warnings.
- False negative hazards.
- Track fragmentation.
- Incorrect spatial associations.
- Stale memory.
- Depth estimation errors.
- Camera calibration issues.

Allow filtering by dataset, event, object type and failure category.

### D. Reproducibility

Show:

- Experiment configuration.
- Model versions.
- Dataset split.
- Sensor configuration.
- Camera calibration.
- Thresholds.
- Evaluation protocol.

The interface should support reproducibility and scientific scrutiny.

---

## 9. Page Five: System

This is the technical diagnostics section.

Display the full processing architecture.

### Components

1. Camera input.
2. Muse Spark / Meta Model API.
3. SAM 3.1 Object Multiplex.
4. Median-depth mask lifting.
5. Persistent world-memory graph.
6. Geometric TTC.
7. SeamlessM4T v2.
8. Ray-Ban Display output.

For each component show:

- Connection status.
- Model/version.
- Processing status.
- Input/output information.
- Error state.
- Latency, if available.

Include sensor calibration and coordinate-frame diagnostics.

### Body-frame validation

Provide a dedicated diagnostic section for:

- Camera coordinate frame.
- World coordinate frame.
- Body coordinate frame.
- Transform consistency.
- Sensor extrinsics.
- Camera intrinsics.

Include the SANPO second-camera validation as a research diagnostic.

The purpose is to demonstrate that the body-frame design was validated against an actual independent camera rather than merely assumed.

---

## 10. Ray-Ban Display HUD Preview

This deserves a dedicated component.

It must faithfully reproduce the actual wearable display rather than looking like a miniature desktop dashboard.

Use the actual device constraints:

- 600 × 600 viewport.
- True black as transparent display background.
- High-contrast accents.
- Large glanceable text.
- No scrolling.
- Minimal content.
- Short actionable messages.

Display a live preview synchronized with the same hazard timeline used by the dashboard.

### HUD states

State 1: No immediate hazard.

Display:

PATH CLEAR

Use a minimal visual indicator.

State 2: Approaching hazard.

Display:

CAUTION LEFT

Include a directional indicator.

State 3: Critical warning.

Display:

STOP

Use the appropriate critical warning treatment.

### Audio output

Show the corresponding spoken directive, limited to six words.

Also show the spatial tone direction.

The HUD preview must update when the timeline is scrubbed.

A user should be able to inspect exactly what the wearer would have seen at a particular timestamp.

Do not turn the wearable display into a conventional navigation map.

---

## 11. Interaction and Motion Design

The interface should feel responsive and polished.

Use purposeful motion:

- Smooth transitions between navigation sections.
- Subtle hover feedback.
- Object highlighting.
- Camera-to-world selection synchronization.
- Timeline playback.
- Memory-state transitions.
- Hazard-state transitions.

The 3D visualization should support natural camera movement.

Avoid unnecessary animations.

Respect reduced-motion preferences.

Do not use animation to simulate actual sensor activity when no live data is available.

---

## 12. Data Integrity and Implementation Requirements

This is a research system, not a fictional dashboard.

Implement a clear distinction between:

- Live data.
- Recorded dataset playback.
- Demonstration data.
- Simulated states.

Never present mock values as real telemetry.

If actual hardware or model APIs are not connected, use a clearly labelled demonstration mode.

Build reusable components and a clean data model.

All major dashboard components should consume a shared event/timeline data source so that camera frames, world memory, hazards and HUD output remain synchronized.

Support:

- Loading states.
- Empty states.
- Connection failures.
- Partial pipeline failures.
- Stale observations.
- Unavailable sensor data.

Use realistic sample data only when clearly marked as illustrative.

Do not invent successful detection or collision-prediction results.

---

## 13. Technical Implementation Preferences

If implementing the frontend from scratch:

- React.
- TypeScript.
- Tailwind CSS.
- Lucide icons.
- Three.js or React Three Fiber for interactive 3D visualization.
- Recharts for scientific charts where appropriate.
- A reusable component architecture.

Suggested components:

- AppShell
- Sidebar
- Topbar
- OverviewDashboard
- WorldMemoryCanvas
- CameraFeed
- ObjectInspector
- HazardPanel
- EventTimeline
- PipelineStatus
- RayBanHUDPreview
- AblationStudio
- DatasetComparison
- FailureAnalysis
- SystemDiagnostics

Keep data handling separate from presentation.

Use a shared event model for synchronized replay.

Make the application responsive, but prioritize a high-quality desktop experience.

---

## 14. Final Quality Bar

The result must look like a polished, credible, premium research product.

Prioritize:

1. A beautiful and scientifically meaningful 3D world-memory visualization.
2. A synchronized live camera feed.
3. An exceptionally clear hazard panel.
4. A genuinely convincing Ray-Ban Display HUD preview.
5. A synchronized temporal event timeline.
6. A rigorous experimental evaluation interface.
7. Consistent Meta-inspired visual design.

The first screen should communicate the entire Locus thesis within five seconds.

A reviewer should immediately see:

- What the camera currently observes.
- What the system remembers beyond the camera frame.
- Which objects are approaching.
- Why a warning was triggered.
- What the wearer receives on the actual glasses.

### The defining design principle

**The camera sees the present. Locus remembers the world.**

Make this principle visible throughout the product.

Do not sacrifice scientific accuracy for visual appeal. Do not sacrifice usability for technical density.

Build an interface that feels simultaneously like a real assistive technology product, a sophisticated spatial computing system and a serious research platform.