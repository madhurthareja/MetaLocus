import { WorldView } from './world3d.js';
import { state, on, currentFrame, selectNode, hoverNode } from './state.js';
import { icon, refreshIcons } from './icons.js';

const LEGEND = [
  { swatch: 'dot', color: '#3E9BFF', label: 'Currently visible' },
  { swatch: 'dot', color: '#4F8A91', label: 'In memory / out of view' },
  { swatch: 'dot', color: '#4F8A91', label: 'Estimated position', ring: true },
  { swatch: 'line', color: '#E69B25', label: 'Predicted trajectory' },
  { swatch: 'dot', color: '#FF8A3D', label: 'Potential hazard' },
];

// Wraps the Three.js WorldView with the chrome docs/UI.md asks for: view-mode
// toggle, legend, and the persistent-memory thesis banner. Used standalone
// on World Memory, and shrunk for the Overview page -- same engine either way.
export class WorldViewPanel {
  constructor(container, { dataset = 'synthetic', compact = false, showBanner = true } = {}) {
    this.container = container;
    container.innerHTML = `
      <div class="world-canvas-wrap ${showBanner ? '' : 'standalone'}">
        <div class="world-stage ${compact ? 'compact' : 'tall'}" id="wv-stage"></div>
        <div class="world-toolbar">
          <div class="btn-group">
            <button class="btn active" data-mode="3d">3D Perspective</button>
            <button class="btn" data-mode="top">Top View</button>
            <button class="btn" data-mode="side">Side View</button>
          </div>
        </div>
        ${showBanner ? `<div class="world-thesis-banner"><strong>The camera sees the present.</strong> Locus remembers the world. Objects stay in this view after they leave the camera frame.</div>` : ''}
        <div class="world-legend">${LEGEND.map((l) => `
          <div class="legend-row"><span class="legend-swatch ${l.swatch}" style="background:${l.color}"></span>${l.label}</div>
        `).join('')}</div>
      </div>`;

    this.stageEl = container.querySelector('#wv-stage');
    this.view = new WorldView(this.stageEl, {
      dataset,
      onSelect: (id) => selectNode(id),
      onHover: (id) => hoverNode(id),
    });

    container.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
      container.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('active', x === b));
      this.view.setViewMode(b.dataset.mode);
    }));

    on('session-loaded', ({ manifest }) => {
      this.view.setDataset(manifest.dataset);
      this.view.frameScene(manifest.timeline, manifest.dataset);
      this._render();
    });
    on('frame-changed', () => this._render());
    on('node-selected', ({ nodeId }) => { this.view.setSelectedNode(nodeId); this._render(); });
    on('node-hovered', ({ nodeId }) => { this.view.setHoveredNode(nodeId); });
    if (state.manifest) this.view.frameScene(state.manifest.timeline, state.manifest.dataset);
    this._render();
  }

  _render() {
    const frame = currentFrame();
    if (frame) this.view.setFrame(frame, state.manifest);
  }
}
