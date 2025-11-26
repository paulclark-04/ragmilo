const messagesEl = document.getElementById('messages');
const form = document.getElementById('question-form');
const textarea = document.getElementById('question');
const clearButton = document.getElementById('clear-convo');

const matiereSelect = document.getElementById('matiere');
const sousMatiereSelect = document.getElementById('sous_matiere');
const enseignantSelect = document.getElementById('enseignant');
const semestreSelect = document.getElementById('semestre');
const promoSelect = document.getElementById('promo');
const thresholdInput = document.getElementById('threshold');
const alphaInput = document.getElementById('alpha');

let isProcessing = false;
let typingIndicator;

function createOption(value, label = value) {
  const option = document.createElement('option');
  option.value = value || '';
  option.textContent = label;
  return option;
}

function populateSelect(select, values) {
  select.innerHTML = '';
  select.appendChild(createOption('', 'Toutes'));
  values.forEach((value) => select.appendChild(createOption(value)));
}

let metadataRecords = [];

async function loadMetadata() {
  try {
    const response = await fetch('/api/metadata');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const metadata = await response.json();
    const unique = metadata.unique || {};
    metadataRecords = metadata.records || [];
    metadataRecords = metadata.records || [];
    populateSelect(matiereSelect, unique.matiere || []);
    populateSelect(sousMatiereSelect, unique.sous_matiere || []);
    populateSelect(enseignantSelect, unique.enseignant || []);
    populateSelect(semestreSelect, unique.semestre || []);
    populateSelect(promoSelect, unique.promo || []);
  } catch (error) {
    console.error('Erreur lors du chargement des métadonnées', error);
  }
}

function filterOptions() {
  const selection = {
    matiere: matiereSelect.value || null,
    sous_matiere: sousMatiereSelect.value || null,
    enseignant: enseignantSelect.value || null,
    semestre: semestreSelect.value || null,
    promo: promoSelect.value || null,
  };

  const computeOptions = (field) => {
    return Array.from(
      new Set(
        metadataRecords
          .filter((rec) =>
            Object.entries(selection).every(([key, value]) =>
              key === field || !value || rec[key] === value
            )
          )
          .map((rec) => rec[field])
          .filter(Boolean)
      )
    );
  };

  const previousMatiere = selection.matiere;
  const matieres = computeOptions('matiere');
  populateSelect(matiereSelect, matieres);
  if (previousMatiere && matieres.includes(previousMatiere)) {
    matiereSelect.value = previousMatiere;
  }
  selection.matiere = matiereSelect.value || null;

  const sousMatieres = computeOptions('sous_matiere');
  const previousSous = selection.sous_matiere;
  populateSelect(sousMatiereSelect, sousMatieres);
  if (previousSous && sousMatieres.includes(previousSous)) {
    sousMatiereSelect.value = previousSous;
  }
  selection.sous_matiere = sousMatiereSelect.value || null;

  const enseignants = computeOptions('enseignant');
  populateSelect(enseignantSelect, enseignants);
  if (selection.enseignant && enseignants.includes(selection.enseignant)) {
    enseignantSelect.value = selection.enseignant;
  }
  selection.enseignant = enseignantSelect.value || null;

  const semestres = computeOptions('semestre');
  const previousSemestre = selection.semestre;
  populateSelect(semestreSelect, semestres);
  if (previousSemestre && semestres.includes(previousSemestre)) {
    semestreSelect.value = previousSemestre;
  }
  selection.semestre = semestreSelect.value || null;

  const promos = computeOptions('promo');
  const previousPromo = selection.promo;
  populateSelect(promoSelect, promos);
  if (previousPromo && promos.includes(previousPromo)) {
    promoSelect.value = previousPromo;
  }
  selection.promo = promoSelect.value || null;
}

function setEmptyState() {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <h3>Bienvenue 👋</h3>
      <p>Pose une question sur l'un de tes cours pour obtenir une réponse sourcée, sans hallucination.</p>
      <div class="loader"></div>
    </div>
  `;
}

function resetConversation() {
  messagesEl.innerHTML = '';
  setEmptyState();
}

function addMessage(role, content, meta = {}) {
  if (messagesEl.querySelector('.empty-state')) {
    messagesEl.innerHTML = '';
  }

  const wrapper = document.createElement('div');
  wrapper.classList.add('message');
  wrapper.classList.add(role === 'user' ? 'message--user' : 'message--assistant');

  if (role === 'assistant') {
    const heading = document.createElement('h3');
    heading.textContent = 'Réponse';
    wrapper.appendChild(heading);
  }

  const text = document.createElement('div');
  text.innerHTML = content.replace(/\n/g, '<br />');
  wrapper.appendChild(text);

  if (role === 'assistant' && meta.sources?.length) {
    const metaBlock = document.createElement('div');
    metaBlock.classList.add('message__meta');
    metaBlock.textContent = `Indice de confiance : ${meta.confidence ?? 'N/A'} | Score top-1 : ${meta.top1 ?? 'N/A'}`;
    wrapper.appendChild(metaBlock);

    const sourceList = document.createElement('div');
    sourceList.classList.add('source-list');
    meta.sources.forEach((source) => {
      const chip = document.createElement('div');
      chip.classList.add('source-chip');
      const label = source.doc_label || source.doc_id;
      chip.textContent = `${label} · p.${source.page} · ${source.score}`;
      chip.title = source.fragment;
      sourceList.appendChild(chip);
    });
    wrapper.appendChild(sourceList);
  }

  messagesEl.appendChild(wrapper);
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

function showTypingIndicator() {
  if (typingIndicator) return;
  typingIndicator = document.createElement('div');
  typingIndicator.className = 'message message--assistant typing-indicator';
  typingIndicator.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;
  messagesEl.appendChild(typingIndicator);
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

function hideTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.remove();
    typingIndicator = null;
  }
}

async function sendQuestion(question) {
  if (isProcessing) return;
  isProcessing = true;

  addMessage('user', question);
  showTypingIndicator();

  const payload = {
    question,
    matiere: matiereSelect.value || null,
    sous_matiere: sousMatiereSelect.value || null,
    enseignant: enseignantSelect.value || null,
    semestre: semestreSelect.value || null,
    promo: promoSelect.value || null,
    threshold: Number(thresholdInput.value) || 0.35,
    alpha: Number(alphaInput.value) || 0.65,
  };

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const stats = data.retrieval_stats || {};

    addMessage('assistant', data.answer || '(Réponse vide)', {
      sources: data.sources || [],
      confidence: data.confidence,
      top1: stats.top1,
    });
  } catch (error) {
    console.error(error);
    addMessage('assistant', "Erreur lors de l'appel à l'API. Vérifie que le serveur est lancé.");
  } finally {
    hideTypingIndicator();
    isProcessing = false;
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;
  textarea.value = '';
  sendQuestion(question);
});

clearButton.addEventListener('click', () => {
  resetConversation();
});

textarea.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.dispatchEvent(new Event('submit'));
  }
});

[matiereSelect, sousMatiereSelect, enseignantSelect, semestreSelect, promoSelect].forEach((select) => {
  select.addEventListener('change', filterOptions);
});

loadMetadata().then(() => {
  filterOptions();
  setEmptyState();
});
