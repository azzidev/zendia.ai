const Chart = {
  canvas: null,
  ctx: null,

  init() {
    this.canvas = document.getElementById('loss-chart');
    this.ctx = this.canvas.getContext('2d');
    this.resize();
    window.addEventListener('resize', () => this.resize());
  },

  resize() {
    const dpr = devicePixelRatio || 1;
    const parent = this.canvas.parentElement;
    this.canvas.width = parent.clientWidth * dpr;
    this.canvas.height = 80 * dpr;
    this.canvas.style.width = parent.clientWidth + 'px';
    this.canvas.style.height = '80px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  },

  draw(lossHistory) {
    const w = this.canvas.parentElement.clientWidth;
    const h = 80;
    this.ctx.clearRect(0, 0, w, h);

    if (lossHistory.length < 2) return;

    const maxL = Math.max(...lossHistory.slice(0, 20), 0.01);
    const step = w / Math.max(lossHistory.length - 1, 1);

    // gradient fill
    const grad = this.ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(255, 118, 117, 0.12)');
    grad.addColorStop(1, 'rgba(255, 118, 117, 0)');

    // area
    this.ctx.beginPath();
    this.ctx.moveTo(0, h - (lossHistory[0] / maxL) * h);
    for (let i = 1; i < lossHistory.length; i++) {
      this.ctx.lineTo(i * step, h - Math.min((lossHistory[i] / maxL) * h, h));
    }
    this.ctx.lineTo((lossHistory.length - 1) * step, h);
    this.ctx.lineTo(0, h);
    this.ctx.closePath();
    this.ctx.fillStyle = grad;
    this.ctx.fill();

    // line
    this.ctx.beginPath();
    this.ctx.moveTo(0, h - (lossHistory[0] / maxL) * h);
    for (let i = 1; i < lossHistory.length; i++) {
      this.ctx.lineTo(i * step, h - Math.min((lossHistory[i] / maxL) * h, h));
    }
    this.ctx.strokeStyle = 'rgba(255, 118, 117, 0.6)';
    this.ctx.lineWidth = 1.5;
    this.ctx.lineJoin = 'round';
    this.ctx.stroke();

    // dot
    const lx = (lossHistory.length - 1) * step;
    const ly = h - Math.min((lossHistory[lossHistory.length - 1] / maxL) * h, h);
    this.ctx.beginPath();
    this.ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    this.ctx.fillStyle = 'rgba(255, 118, 117, 0.9)';
    this.ctx.fill();
  }
};
