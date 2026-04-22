const Network3D = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  layerMeshes: [],
  particleSystems: [],
  connectionLines: [],
  built: false,

  init() {
    const wrap = document.getElementById('scene-wrap');

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x08080d);
    this.scene.fog = new THREE.FogExp2(0x08080d, 0.008);

    this.camera = new THREE.PerspectiveCamera(60, wrap.clientWidth / wrap.clientHeight, 0.1, 2000);
    this.camera.position.set(0, 80, 250);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    this.renderer.setPixelRatio(devicePixelRatio);
    wrap.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.autoRotate = false;

    // luzes
    const ambient = new THREE.AmbientLight(0x333355, 0.5);
    this.scene.add(ambient);
    const point = new THREE.PointLight(0x6c5ce7, 1, 500);
    point.position.set(0, 100, 100);
    this.scene.add(point);
    const point2 = new THREE.PointLight(0x00cec9, 0.5, 500);
    point2.position.set(0, -50, -100);
    this.scene.add(point2);

    // grid no chão
    const grid = new THREE.GridHelper(600, 40, 0x1a1a2e, 0x111122);
    grid.position.y = -60;
    this.scene.add(grid);

    window.addEventListener('resize', () => {
      this.camera.aspect = wrap.clientWidth / wrap.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    });
  },

  buildNetwork(arch) {
    if (this.built || !arch || arch.length === 0) return;

    // limpa anterior
    this.layerMeshes.forEach(m => this.scene.remove(m));
    this.connectionLines.forEach(m => this.scene.remove(m));
    this.particleSystems.forEach(m => this.scene.remove(m));
    this.layerMeshes = [];
    this.connectionLines = [];
    this.particleSystems = [];

    const n = arch.length;
    const spacing = 80;
    const startX = -(n - 1) * spacing / 2;

    for (let l = 0; l < n; l++) {
      const count = arch[l];
      const x = startX + l * spacing;

      // calcula grid
      const cols = Math.ceil(Math.sqrt(count * 1.5));
      const rows = Math.ceil(count / cols);
      const cellSize = Math.min(0.8, 40 / Math.max(cols, rows));

      // partículas pra cada neurônio
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);

      for (let i = 0; i < count; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        positions[i * 3] = x;
        positions[i * 3 + 1] = (row - rows / 2) * cellSize;
        positions[i * 3 + 2] = (col - cols / 2) * cellSize;

        // cor inicial
        if (l === 0) { colors[i*3]=0.42; colors[i*3+1]=0.36; colors[i*3+2]=0.91; }
        else if (l === n-1) { colors[i*3]=0.99; colors[i*3+1]=0.8; colors[i*3+2]=0.43; }
        else { colors[i*3]=0.05; colors[i*3+1]=0.12; colors[i*3+2]=0.25; }
      }

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: cellSize * 1.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: true,
      });

      const points = new THREE.Points(geometry, material);
      this.scene.add(points);
      this.particleSystems.push(points);

      // label (sprite de texto)
      const label = this.makeLabel(
        l === 0 ? 'EMBED' : (l === n-1 ? 'OUTPUT' : 'LAYER ' + l),
        count.toLocaleString() + ' neurons',
        l === 0 ? '#a29bfe' : (l === n-1 ? '#fdcb6e' : '#00cec9')
      );
      label.position.set(x, rows * cellSize / 2 + 8, 0);
      this.scene.add(label);
      this.layerMeshes.push(label);

      // conexão com próxima camada
      if (l < n - 1) {
        const nextX = startX + (l + 1) * spacing;
        const lineGeo = new THREE.BufferGeometry();
        const linePositions = new Float32Array(6 * 6); // 6 linhas
        for (let i = 0; i < 6; i++) {
          const t = (i / 5) * 2 - 1;
          const spread = Math.min(rows * cellSize / 2, 15);
          linePositions[i*6] = x + 2;
          linePositions[i*6+1] = t * spread;
          linePositions[i*6+2] = 0;
          linePositions[i*6+3] = nextX - 2;
          linePositions[i*6+4] = t * spread;
          linePositions[i*6+5] = 0;
        }
        lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        const lineMat = new THREE.LineBasicMaterial({ color: 0x6c5ce7, transparent: true, opacity: 0.1 });
        const lines = new THREE.LineSegments(lineGeo, lineMat);
        this.scene.add(lines);
        this.connectionLines.push(lines);
      }
    }

    this.built = true;
  },

  makeLabel(title, subtitle, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color;
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(title, 128, 24);
    ctx.fillStyle = '#888';
    ctx.font = '14px monospace';
    ctx.fillText(subtitle, 128, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(30, 8, 1);
    return sprite;
  },

  updateActivations(activations) {
    if (!activations || activations.length === 0 || this.particleSystems.length === 0) return;

    // activations vem como array de {name, size, activations[], min, max}
    // mapeia pro particleSystem correspondente (offset de 1 porque index 0 é embed)
    for (let a = 0; a < activations.length; a++) {
      const snap = activations[a];
      // cada 2 snapshots = 1 block (attn + ffn), mapeia pro particleSystem index
      const layerIdx = Math.floor(a / 2) + 1; // +1 pula embed
      if (layerIdx >= this.particleSystems.length) continue;

      const ps = this.particleSystems[layerIdx];
      const colors = ps.geometry.attributes.color;
      const range = snap.max - snap.min || 1;

      for (let i = 0; i < Math.min(snap.activations.length, colors.count); i++) {
        const t = (snap.activations[i] - snap.min) / range; // 0-1

        // heatmap: azul escuro → ciano → amarelo → vermelho
        let r, g, b;
        if (t < 0.25) {
          r = 0.05; g = 0.05 + t * 2; b = 0.3 + t * 2;
        } else if (t < 0.5) {
          const s = (t - 0.25) * 4;
          r = s * 0.2; g = 0.55 + s * 0.25; b = 0.8 - s * 0.3;
        } else if (t < 0.75) {
          const s = (t - 0.5) * 4;
          r = 0.2 + s * 0.7; g = 0.8 - s * 0.1; b = 0.5 - s * 0.4;
        } else {
          const s = (t - 0.75) * 4;
          r = 0.9 + s * 0.1; g = 0.7 - s * 0.5; b = 0.1;
        }

        colors.setXYZ(i, r, g, b);
      }
      colors.needsUpdate = true;
    }
  },

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
};
