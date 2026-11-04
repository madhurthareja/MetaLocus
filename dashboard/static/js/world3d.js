import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { hazardSeverity } from './state.js';

// The signature visualization: a persistent world-coordinate memory graph,
// rendered so the difference between "currently observed", "remembered but
// out of frame" and "predicted" is impossible to miss. See docs/UI.md
// section 5.A -- this is deliberately not a static illustration: real
// rotate/zoom/select, real per-node state driving color/opacity/geometry.
//
// Coordinate note: SANPO's world is right-handed and Z-up (measured -- see
// sanpo_loader.py's rotation-convention note); Three.js is right-handed and
// Y-up. (x, y, z) -> (x, z, -y) is the proper rotation between the two. A
// plain Y/Z swap would be a reflection: the scene would render as the
// mirror image of the real street, objects on the wearer's right drawn on
// their left. Every other dataset maps straight through.
function worldToThree(p, dataset, out = new THREE.Vector3()) {
  if (dataset === 'sanpo') return out.set(p[0], p[2], -p[1]);
  return out.set(p[0], p[1], p[2]);
}

const COLORS = {
  bg: 0x101820,
  grid: 0x1c2832,
  visible: 0x3e9bff,
  remembered: 0x4f8a91,
  predicted: 0xe69b25,
  hazard: 0xff8a3d,
  critical: 0xff5c5c,
  wearer: 0xffffff,
  text: 0xc7d4dc,
};

const PREDICT_HORIZON_S = 3.0;
const MIN_SPEED_FOR_TRAJECTORY = 0.15; // m/s -- below this, velocity is noise, not real motion
const STALE_CONFIDENCE_FOR_ESTIMATE = 0.45; // below this, a node is shown as "estimated" (dashed, uncertainty halo)

export class WorldView {
  constructor(container, { dataset = 'synthetic', onSelect = () => {}, onHover = () => {} } = {}) {
    this.container = container;
    this.dataset = dataset;
    this.onSelect = onSelect;
    this.onHover = onHover;
    this.mode = '3d'; // '3d' | 'top' | 'side'
    this.selectedNodeId = null;
    this.hoveredNodeId = null;
    this.filter = { state: 'all', concept: 'all' }; // 'all' | 'visible' | 'memory'
    this._nodeMeshes = new Map(); // node_id -> { group, box, haloOrLine }
    this._labelEls = new Map(); // node_id -> HTMLElement

    this._initScene();
    this._initLabelLayer();
    this._bindInteraction();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
    new ResizeObserver(() => this._onResize()).observe(container);
  }

  setDataset(dataset) {
    this.dataset = dataset;
  }

  _initScene() {
    const { clientWidth: w, clientHeight: h } = this.container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.bg);
    this.scene.fog = new THREE.Fog(COLORS.bg, 18, 48);

    this.perspCamera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500);
    this.perspCamera.position.set(6, 7, 10);
    const orthoSize = 14;
    this.orthoCamera = new THREE.OrthographicCamera(-orthoSize, orthoSize, orthoSize / (w / h), -orthoSize / (w / h), 0.1, 500);
    this.camera = this.perspCamera;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 60;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    const hemi = new THREE.HemisphereLight(0x3a6a8a, 0x101820, 1.1);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.35);
    dir.position.set(5, 10, 3);
    this.scene.add(dir);

    const grid = new THREE.GridHelper(80, 40, COLORS.grid, COLORS.grid);
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    this.scene.add(grid);

    this.nodeGroup = new THREE.Group();
    this.scene.add(this.nodeGroup);
    this.wearerGroup = new THREE.Group();
    this.scene.add(this.wearerGroup);

    this._buildWearerRig();

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
  }

  _buildWearerRig() {
    const body = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.22, 0.5, 4, 8),
      new THREE.MeshBasicMaterial({ color: COLORS.wearer, transparent: true, opacity: 0.85 })
    );
    body.position.y = 0.6;
    this.wearerGroup.add(body);

    // Camera field-of-view frustum -- a diagrammatic ~60 degree cone along
    // the real pose orientation, not a precise intrinsics-exact frustum
    // (that would need per-frame projection work the view doesn't need);
    // honestly approximate, not fabricated telemetry.
    const depth = 3.2;
    const halfAngle = (58 * Math.PI) / 180 / 2;
    const half = Math.tan(halfAngle) * depth;
    const pts = [
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(-half, -half * 0.75, depth),
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(half, -half * 0.75, depth),
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(-half, half * 0.75, depth),
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(half, half * 0.75, depth),
      new THREE.Vector3(-half, -half * 0.75, depth), new THREE.Vector3(half, -half * 0.75, depth),
      new THREE.Vector3(half, -half * 0.75, depth), new THREE.Vector3(half, half * 0.75, depth),
      new THREE.Vector3(half, half * 0.75, depth), new THREE.Vector3(-half, half * 0.75, depth),
      new THREE.Vector3(-half, half * 0.75, depth), new THREE.Vector3(-half, -half * 0.75, depth),
    ];
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    this.frustum = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: 0x3e9bff, transparent: true, opacity: 0.55 }));
    this.frustum.position.y = 0.75;
    this.wearerGroup.add(this.frustum);
  }

  _initLabelLayer() {
    this.labelLayer = document.createElement('div');
    this.labelLayer.style.cssText = 'position:absolute; inset:0; pointer-events:none; overflow:hidden;';
    this.container.style.position = 'relative';
    this.container.appendChild(this.labelLayer);
  }

  _bindInteraction() {
    const dom = this.renderer.domElement;
    dom.style.cursor = 'grab';
    dom.addEventListener('click', (ev) => {
      const hit = this._pick(ev);
      this.onSelect(hit ? hit.nodeId : null);
    });
    dom.addEventListener('mousemove', (ev) => {
      const hit = this._pick(ev);
      dom.style.cursor = hit ? 'pointer' : 'grab';
      this.onHover(hit ? hit.nodeId : null);
    });
  }

  _pick(ev) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const meshes = [];
    this._nodeMeshes.forEach((m) => meshes.push(m.hitbox));
    const hits = this.raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    return { nodeId: hits[0].object.userData.nodeId };
  }

  setViewMode(mode) {
    // Top <-> side is a position/up change on the SAME (ortho) camera, so it
    // tweens smoothly. 3D <-> ortho swaps the camera object itself (different
    // projection, no sane way to interpolate) -- that transition gets a very
    // short opacity dip instead, so the cut reads as a deliberate mode change
    // rather than a jarring frame-to-frame jump.
    const crossingProjection = (this.mode === '3d') !== (mode === '3d');
    this.mode = mode;

    const applyMode = () => {
      if (mode === '3d') {
        this.camera = this.perspCamera;
        this.controls.object = this.camera;
        this.controls.enableRotate = true;
        this.controls.update();
        return;
      }
      this.camera = this.orthoCamera;
      this.controls.object = this.camera;
      this.controls.enableRotate = false;
      const targetPos = mode === 'top' ? new THREE.Vector3(0, 30, 0.01) : new THREE.Vector3(30, 1.5, 0);
      const targetUp = mode === 'top' ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0);
      if (crossingProjection) {
        this.orthoCamera.position.copy(targetPos);
        this.orthoCamera.up.copy(targetUp);
        this.controls.update();
      } else {
        this._camTween = { cam: this.orthoCamera, startPos: this.orthoCamera.position.clone(), startUp: this.orthoCamera.up.clone(), targetPos, targetUp, t0: performance.now(), duration: 420 };
      }
    };

    if (crossingProjection) {
      const dom = this.renderer.domElement;
      dom.style.transition = 'opacity 0.15s ease';
      dom.style.opacity = '0';
      setTimeout(() => { applyMode(); dom.style.opacity = '1'; }, 160);
    } else {
      applyMode();
    }
  }

  // Scenes vary wildly in scale (a synthetic demo a few meters across vs. a
  // real SANPO session spanning tens of meters) -- a fixed starting camera
  // would leave most sessions either empty-looking or all fog. Frame the
  // starting view around the wearer's full trajectory and every node's
  // position the first time a session loads.
  frameScene(timeline, dataset) {
    const pts = [];
    timeline.forEach((rec) => {
      pts.push(worldToThree(rec.wearer_pos, dataset));
      rec.nodes.forEach((n) => pts.push(worldToThree(n.p_world, dataset)));
    });
    if (!pts.length) return;
    const min = pts[0].clone(), max = pts[0].clone();
    pts.forEach((p) => { min.min(p); max.max(p); });
    const center = min.clone().add(max).multiplyScalar(0.5);
    let radius = 2;
    pts.forEach((p) => { radius = Math.max(radius, p.distanceTo(center)); });

    // A bounding SPHERE, not box, guarantees every point fits inside the
    // frustum regardless of which direction the scene happens to be long
    // in -- fitting an oriented box to a fixed-angle diagonal view (the
    // first version of this) left real content outside the frame.
    const vFov = (this.perspCamera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.perspCamera.aspect);
    const limitingHalfAngle = Math.min(vFov, hFov) / 2;
    const distance = (radius / Math.sin(limitingHalfAngle)) * 1.25;

    const dir = new THREE.Vector3(0.55, 0.5, 0.75).normalize();
    this.perspCamera.position.copy(center).addScaledVector(dir, distance);
    this.controls.target.copy(center);
    this.controls.minDistance = Math.max(1, radius * 0.15);
    this.controls.maxDistance = distance * 5;
    this.scene.fog.near = distance * 0.8;
    this.scene.fog.far = distance * 3.5 + radius;
    this.controls.update();
  }

  setSelectedNode(id) {
    if (id != null && id !== this.selectedNodeId) {
      const rec = this._nodeMeshes.get(id);
      if (rec) rec.selectPopT0 = performance.now();
    }
    this.selectedNodeId = id;
  }
  setHoveredNode(id) { this.hoveredNodeId = id; }
  setFilter(filter) {
    this.filter = { ...this.filter, ...filter };
    if (this.lastFrame) this.setFrame(this.lastFrame, this.lastManifest);
  }

  // ---- per-frame update --------------------------------------------------

  setFrame(frame, manifest) {
    if (!frame) return;
    this.lastFrame = frame;
    this.lastManifest = manifest;

    const wearerPos = worldToThree(frame.wearer_pos, this.dataset);
    this.wearerGroup.position.copy(wearerPos);
    if (frame.camera_R_world_cam) {
      const R = frame.camera_R_world_cam;
      const forwardWorld = [R[0][2], R[1][2], R[2][2]]; // local +Z forward (OpenCV convention)
      const fwd3 = worldToThree(forwardWorld, this.dataset).sub(worldToThree([0, 0, 0], this.dataset)).normalize();
      const upWorld = [-R[0][1], -R[1][1], -R[2][1]]; // local -Y (OpenCV +Y is down)
      const up3 = worldToThree(upWorld, this.dataset).sub(worldToThree([0, 0, 0], this.dataset)).normalize();
      const m = new THREE.Matrix4();
      const right3 = new THREE.Vector3().crossVectors(fwd3, up3).normalize();
      const trueUp = new THREE.Vector3().crossVectors(right3, fwd3).normalize();
      m.makeBasis(right3, trueUp, fwd3);
      this.wearerGroup.quaternion.setFromRotationMatrix(m);
    }

    const liveIds = new Set();
    for (const node of frame.nodes) {
      liveIds.add(node.node_id);
      this._updateNodeMesh(node);
    }
    for (const [nodeId, mesh] of this._nodeMeshes) {
      if (!liveIds.has(nodeId)) this._removeNodeMesh(nodeId);
    }
  }

  _removeNodeMesh(nodeId) {
    const m = this._nodeMeshes.get(nodeId);
    if (m) {
      this.nodeGroup.remove(m.group);
      m.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    }
    this._nodeMeshes.delete(nodeId);
    const label = this._labelEls.get(nodeId);
    if (label) label.remove();
    this._labelEls.delete(nodeId);
  }

  _ensureNodeMesh(nodeId, concept) {
    if (this._nodeMeshes.has(nodeId)) return this._nodeMeshes.get(nodeId);

    const group = new THREE.Group();
    const boxGeom = new THREE.BoxGeometry(1, 1, 1);
    const edgesGeom = new THREE.EdgesGeometry(boxGeom);
    const box = new THREE.LineSegments(edgesGeom, new THREE.LineBasicMaterial({ color: COLORS.visible }));
    const fill = new THREE.Mesh(boxGeom, new THREE.MeshBasicMaterial({ color: COLORS.visible, transparent: true, opacity: 0.12 }));
    group.add(box, fill);

    const haloGeom = new THREE.SphereGeometry(1, 16, 12);
    const halo = new THREE.Mesh(haloGeom, new THREE.MeshBasicMaterial({ color: COLORS.remembered, transparent: true, opacity: 0, wireframe: true }));
    group.add(halo);

    const ringGeom = new THREE.RingGeometry(0.85, 1, 24);
    const ring = new THREE.Mesh(ringGeom, new THREE.MeshBasicMaterial({ color: COLORS.hazard, transparent: true, opacity: 0, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    group.add(ring);

    // Selection feedback -- deliberately a separate mesh from the hazard
    // ring above so "I clicked this" never reads as "this is a hazard".
    const selGeom = new THREE.BoxGeometry(1.25, 1.25, 1.25);
    const selectGlow = new THREE.LineSegments(new THREE.EdgesGeometry(selGeom), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
    group.add(selectGlow);

    const trajGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const traj = new THREE.Line(trajGeom, new THREE.LineDashedMaterial({ color: COLORS.predicted, dashSize: 0.25, gapSize: 0.15, transparent: true, opacity: 0 }));
    group.add(traj);

    const hitbox = new THREE.Mesh(new THREE.SphereGeometry(0.9, 8, 6), new THREE.MeshBasicMaterial({ visible: false }));
    hitbox.userData.nodeId = nodeId;
    group.add(hitbox);

    this.nodeGroup.add(group);

    const label = document.createElement('div');
    label.style.cssText = `
      position:absolute; transform: translate(-50%, -140%); font-family: 'JetBrains Mono', monospace;
      font-size: 10px; color: #C7D4DC; background: rgba(16,24,32,0.72); padding: 2px 6px;
      border-radius: 4px; white-space: nowrap; pointer-events: none; border: 1px solid rgba(255,255,255,0.08);`;
    this.labelLayer.appendChild(label);

    const rec = { group, box, fill, halo, ring, selectGlow, traj, hitbox, label, selectPopT0: null };
    this._nodeMeshes.set(nodeId, rec);
    return rec;
  }

  _updateNodeMesh(node) {
    const rec = this._ensureNodeMesh(node.node_id, node.concept);

    const stateOk = this.filter.state === 'all' || (this.filter.state === 'visible' && node.visible) || (this.filter.state === 'memory' && !node.visible);
    const conceptOk = this.filter.concept === 'all' || this.filter.concept === node.concept;
    const shown = stateOk && conceptOk;
    rec.group.visible = shown;
    rec.label.style.display = shown ? '' : 'none';
    if (!shown) return;

    const pos = worldToThree(node.p_world, this.dataset);
    rec.group.position.copy(pos);

    const ext = node.extent && node.extent.some((v) => v > 0.02) ? node.extent : [0.5, 0.8, 0.5];
    const sx = Math.max(0.3, Math.abs(ext[0]));
    const sy = Math.max(0.3, Math.abs(this.dataset === 'sanpo' ? ext[2] : ext[1]));
    const sz = Math.max(0.3, Math.abs(this.dataset === 'sanpo' ? ext[1] : ext[2]));
    rec.box.scale.set(sx, sy, sz);
    rec.fill.scale.set(sx, sy, sz);
    rec.group.position.y += sy / 2;

    const isSelected = this.selectedNodeId === node.node_id;
    const isHovered = this.hoveredNodeId === node.node_id;
    const isStaleEstimate = !node.visible && node.confidence < STALE_CONFIDENCE_FOR_ESTIMATE;
    const edge = this.lastFrame?.edges.find((e) => e.node_id === node.node_id);
    const edgeSev = hazardSeverity(edge);
    const sev = edgeSev === 'critical' || edgeSev === 'hazard' ? edgeSev : null;

    let color = COLORS.visible;
    let opacity = 0.9;
    if (!node.visible) {
      color = COLORS.remembered;
      opacity = 0.25 + 0.45 * node.confidence; // gradual fade as memory goes stale
    }

    rec.box.material.color.setHex(color);
    rec.box.material.transparent = true;
    rec.box.material.opacity = opacity;
    rec.fill.material.color.setHex(color);
    rec.fill.material.opacity = node.visible ? 0.14 : 0.05 * node.confidence;
    rec.box.material.dashed = false;

    // "Estimated position": growing uncertainty halo once memory is stale
    // enough that the last known point is a genuine estimate, not a fresh
    // observation -- visualizes confidence decay, never claims a new fact.
    if (isStaleEstimate) {
      const haloScale = 1.0 + (1 - node.confidence) * 1.8;
      rec.halo.scale.set(sx * haloScale, sy * haloScale, sz * haloScale);
      rec.halo.material.opacity = 0.18 * (1 - node.confidence) + 0.05;
    } else {
      rec.halo.material.opacity = 0;
    }

    // Predicted trajectory: a short dashed extrapolation along current
    // velocity, drawn only for genuinely-moving nodes -- distinct geometry
    // (a line, not a box) so a prediction can never be read as an observation.
    const speed = Math.hypot(...node.velocity_world);
    if (speed > MIN_SPEED_FOR_TRAJECTORY) {
      const v3 = worldToThree(node.velocity_world, this.dataset).sub(worldToThree([0, 0, 0], this.dataset));
      const end = v3.clone().multiplyScalar(PREDICT_HORIZON_S);
      rec.traj.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), end]);
      rec.traj.computeLineDistances();
      rec.traj.material.opacity = 0.75;
    } else {
      rec.traj.material.opacity = 0;
    }

    // Potential/active hazard ring, independent of the box's own state.
    if (sev) {
      rec.ring.material.color.setHex(sev === 'critical' ? COLORS.critical : COLORS.hazard);
      rec.ring.material.opacity = 0.8;
      rec.ring.scale.setScalar(1 + 0.15 * Math.sin(performance.now() / 300));
    } else {
      rec.ring.material.opacity = 0;
    }

    if (isSelected) {
      let popScale = 1;
      if (rec.selectPopT0 != null) {
        const dt = performance.now() - rec.selectPopT0;
        const dur = 300;
        if (dt < dur) popScale = 1 + 0.35 * (1 - dt / dur);
        else rec.selectPopT0 = null;
      }
      rec.selectGlow.scale.set(sx * popScale, sy * popScale, sz * popScale);
      rec.selectGlow.material.opacity = 0.5 + 0.25 * Math.sin(performance.now() / 220);
      rec.box.material.color.setHex(0xffffff);
      rec.box.material.opacity = 1;
    } else {
      rec.selectGlow.material.opacity = 0;
      if (isHovered) {
        // Brighten toward white rather than snapping fully white, so a
        // casual hover reads as "lighter touch" than a committed selection.
        rec.box.material.color.lerp(new THREE.Color(0xffffff), 0.45);
        rec.box.material.opacity = Math.min(1, opacity + 0.3);
      }
    }

    rec.label.textContent = `#${node.node_id} ${node.concept}`;
    rec.label.style.opacity = node.visible || isSelected || isHovered ? '1' : '0.55';
  }

  // ---- render loop --------------------------------------------------------

  _animate() {
    requestAnimationFrame(this._animate);
    if (this._camTween) {
      const { cam, startPos, startUp, targetPos, targetUp, t0, duration } = this._camTween;
      const t = Math.min(1, (performance.now() - t0) / duration);
      const e = 1 - Math.pow(1 - t, 3); // ease-out cubic
      cam.position.lerpVectors(startPos, targetPos, e);
      cam.up.lerpVectors(startUp, targetUp, e);
      if (t >= 1) this._camTween = null;
    }
    this.controls.update();
    this._updateLabels();
    this.renderer.render(this.scene, this.camera);
  }

  _updateLabels() {
    const { clientWidth: w, clientHeight: h } = this.container;
    const v = new THREE.Vector3();
    this._nodeMeshes.forEach((rec) => {
      if (!rec.group.visible) { rec.label.style.display = 'none'; return; }
      rec.group.getWorldPosition(v);
      v.project(this.camera);
      if (v.z > 1) { rec.label.style.display = 'none'; return; }
      const x = (v.x * 0.5 + 0.5) * w;
      const y = (-v.y * 0.5 + 0.5) * h;
      rec.label.style.display = 'block';
      rec.label.style.left = `${x}px`;
      rec.label.style.top = `${y}px`;
    });
  }

  _onResize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.perspCamera.aspect = w / h;
    this.perspCamera.updateProjectionMatrix();
    const orthoSize = 14;
    this.orthoCamera.left = -orthoSize; this.orthoCamera.right = orthoSize;
    this.orthoCamera.top = orthoSize / (w / h); this.orthoCamera.bottom = -orthoSize / (w / h);
    this.orthoCamera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  dispose() {
    this.renderer.dispose();
  }
}
