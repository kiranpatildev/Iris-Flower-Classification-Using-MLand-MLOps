const presets = {
  setosa: [5.1, 3.5, 1.4, 0.2],
  versicolor: [6.0, 2.9, 4.5, 1.5],
  virginica: [6.7, 3.0, 5.2, 2.3],
};
const names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
const form = document.querySelector('#prediction-form');
const output = document.querySelector('#result-content');
const submit = form.querySelector('button[type="submit"]');

document.querySelectorAll('[data-preset]').forEach(button => {
  button.addEventListener('click', () => {
    names.forEach((name, i) => { form.elements[name].value = presets[button.dataset.preset][i]; });
    form.requestSubmit();
  });
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const measurements = Object.fromEntries(names.map(name => [name, Number(form.elements[name].value)]));
  submit.disabled = true;
  submit.firstElementChild.textContent = 'Analyzing flower…';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(measurements),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Please check the measurements and try again.');
    const details = {
      setosa: 'Small petals and broad sepals are a classic Setosa signature.',
      versicolor: 'Balanced petal proportions point toward Versicolor.',
      virginica: 'Longer, wider petals are typical of Virginica.',
    };
    const rows = Object.entries(result.probabilities).map(([species, score]) => `
      <div class="prob-row"><span>${species}</span><span class="bar"><span style="width:${(score * 100).toFixed(1)}%"></span></span><strong>${(score * 100).toFixed(1)}%</strong></div>`).join('');
    output.innerHTML = `<div class="prediction"><div class="prediction-top"><div class="flower-art" aria-hidden="true"><span class="petal p1"></span><span class="petal p2"></span><span class="petal p3"></span><span class="petal p4"></span><span class="petal p5"></span><span class="flower-center"></span><span class="stem"></span><span class="leaf l1"></span><span class="leaf l2"></span></div><div><span class="prediction-label">MOST LIKELY SPECIES</span><h3>Iris ${result.species}</h3><p class="prediction-copy">${details[result.species]}</p></div></div><div class="probabilities" aria-label="Class probabilities">${rows}</div></div>`;
  } catch (error) {
    output.innerHTML = `<p class="error-state" role="alert"></p>`;
    output.querySelector('.error-state').textContent = error.message;
  } finally {
    submit.disabled = false;
    submit.firstElementChild.textContent = 'Identify flower';
  }
});

