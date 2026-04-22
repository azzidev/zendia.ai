const Heatmap = {
  canvas: null,
  ctx: null,

  init() {
    this.canvas = document.getElementById('heatmap');
    this.ctx = this.canvas.getContext('2d');
  },

  draw(activations) {
    if (!activations || activations.length === 0) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    const numLayers = activations.length;
    const rowH = Math.floor(h / numLayers);

    for (let l = 0; l < numLayers; l++) {
      const snap = activations[l];
      const acts = snap.activations;
      if (!acts || acts.length === 0) continue;

      const range = snap.max - snap.min || 1;
      const pixelW = w / acts.length;
      const y = l * rowH;

      for (let i = 0; i < acts.length; i++) {
        const t = (acts[i] - snap.min) / range;
        this.ctx.fillStyle = this.heatColor(t);
        this.ctx.fillRect(Math.floor(i * pixelW), y, Math.ceil(pixelW) + 1, rowH - 1);
      }

      // label
      this.ctx.fillStyle = 'rgba(255,255,255,0.6)';
      this.ctx.font = '9px monospace';
      this.ctx.textBaseline = 'top';
      this.ctx.fillText(snap.name, 3, y + 2);
    }
  },

  heatColor(t) {
    let r, g, b;
    if (t < 0.25) {
      r = 8; g = Math.round(8 + t * 400); b = Math.round(40 + t * 600);
    } else if (t < 0.5) {
      const s = (t - 0.25) * 4;
      r = Math.round(50 * s); g = Math.round(108 + s * 100); b = Math.round(190 - s * 60);
    } else if (t < 0.75) {
      const s = (t - 0.5) * 4;
      r = Math.round(50 + s * 200); g = Math.round(208 - s * 30); b = Math.round(130 - s * 100);
    } else {
      const s = (t - 0.75) * 4;
      r = Math.round(250); g = Math.round(178 - s * 140); b = Math.round(30 - s * 30);
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
};
