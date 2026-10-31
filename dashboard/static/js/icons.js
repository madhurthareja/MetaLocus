// Thin wrapper around the real Lucide icon library (loaded via CDN in
// index.html as a global `lucide`). Components emit `<i data-lucide="...">`
// placeholders in their template strings and call refreshIcons() once after
// writing to the DOM; Lucide swaps each placeholder for inline SVG and
// leaves already-converted ones alone, so calling this liberally is safe.

export function icon(name, extra = '') {
  return `<i data-lucide="${name}" ${extra}></i>`;
}

export function refreshIcons() {
  requestAnimationFrame(() => {
    if (window.lucide) window.lucide.createIcons();
  });
}
