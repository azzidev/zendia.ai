Network3D.init();
Heatmap.init();
Chart.init();
WS.connect();

WS.onUpdate((state, data) => {
  // métricas
  document.getElementById('v-epoch').textContent = state.epoch;
  document.getElementById('v-step').textContent = state.step ? state.step.toLocaleString() : '0';
  document.getElementById('v-loss').textContent = state.loss < 0.001
    ? state.loss.toExponential(2) : state.loss.toFixed(4);
  document.getElementById('v-avg').textContent = state.avgLoss ? state.avgLoss.toFixed(4) : '—';
  document.getElementById('v-tokens').textContent = state.tokensSeen
    ? state.tokensSeen.toLocaleString() : '0';
  document.getElementById('v-phase').textContent = state.phase || '—';

  // arquitetura
  if (state.architecture && state.architecture.length > 0) {
    document.getElementById('arch-pills').innerHTML = state.architecture.map((s, i) =>
      (i > 0 ? '<span class="arch-arrow">→</span>' : '') +
      '<span class="arch-pill">' + s.toLocaleString() + '</span>'
    ).join('');

    // constrói rede 3D na primeira vez
    Network3D.buildNetwork(state.architecture);
  }

  // ativações
  if (state.activations && state.activations.length > 0) {
    Network3D.updateActivations(state.activations);
    Heatmap.draw(state.activations);
  }

  // sample
  if (state.sample) {
    document.getElementById('sample-text').textContent = state.sample;
  }

  // loss chart
  Chart.draw(WS.lossHistory);
});

// render loop
function animate() {
  Network3D.animate();
  requestAnimationFrame(animate);
}
animate();
