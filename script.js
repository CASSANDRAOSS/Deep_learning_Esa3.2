let vocab = [];
let word2idx = {};
let idx2word = {};
let sequences = [];
let nextWords = [];
const sequenceLength = 5;

let model;
let modelReady = false;
let autoInterval;

let trainingChart;
let lossValues = [];
let accuracyValues = [];
let epochLabels = [];

// Buttons & UI-Elemente
const predictBtn = document.getElementById("predictBtn");
const nextBtn = document.getElementById("nextBtn");
const autoBtn = document.getElementById("autoBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");
const trainingStatusDiv = document.getElementById("trainingStatus");

function setButtonsEnabled(enabled) {
  [predictBtn, nextBtn, autoBtn, stopBtn, resetBtn].forEach((btn) => {
    btn.disabled = !enabled;
  });
}

// ------------------------------------------------------------
// HILFSFUNKTIONEN
// ------------------------------------------------------------
function cleanText(text) {
  return text
    .toLowerCase()
    .replace(/[.,!?;:"„“()\-\n]/g, " ")
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0);
}

function showInputWarning(message) {
  const predDiv = document.getElementById("predictions");
  predDiv.innerHTML = `<div class="warning-box">${message}</div>`;
}

function clearPredictions() {
  document.getElementById("predictions").innerHTML = "";
}

// ------------------------------------------------------------
// DATEN LADEN
// ------------------------------------------------------------
async function loadData() {
  try {
    const response = await fetch("data/3.text_corpus.txt");

    if (!response.ok) {
      throw new Error("Trainingsdatei konnte nicht geladen werden.");
    }

    const text = await response.text();
    const words = cleanText(text);

    if (words.length <= sequenceLength) {
      throw new Error("Der Trainingskorpus ist zu klein.");
    }

    vocab = ["<pad>", "<unk>", ...Array.from(new Set(words))];

    vocab.forEach((word, idx) => {
      word2idx[word] = idx;
      idx2word[idx] = word;
    });

    for (let i = 0; i <= words.length - sequenceLength - 1; i++) {
      const seq = words.slice(i, i + sequenceLength);
      sequences.push(seq.map((w) => word2idx[w]));
      nextWords.push(word2idx[words[i + sequenceLength]]);
    }

    console.log(`Vokabulargröße: ${vocab.length}`);
    console.log(`Anzahl der Sequenzen: ${sequences.length}`);
  } catch (error) {
    console.error(error);
    statusDiv.textContent = "Fehler beim Laden der Trainingsdaten.";

    if (trainingStatusDiv) {
      trainingStatusDiv.textContent =
        "Die Anwendung konnte nicht gestartet werden.";
    }

    showInputWarning(error.message);
    throw error;
  }
}

// ------------------------------------------------------------
// TRAININGSDATEN VORBEREITEN
// ------------------------------------------------------------
function prepareTrainingData() {
  const X = [];
  const y = [];

  sequences.forEach((seq, i) => {
    const xSeq = seq.map((idx) => {
      const oneHot = new Array(vocab.length).fill(0);
      oneHot[idx] = 1;
      return oneHot;
    });

    X.push(xSeq);

    const yVec = new Array(vocab.length).fill(0);
    yVec[nextWords[i]] = 1;
    y.push(yVec);
  });

  return {
    X_tensor: tf.tensor3d(X),
    y_tensor: tf.tensor2d(y),
  };
}

// ------------------------------------------------------------
// MODELL ERSTELLEN
// ------------------------------------------------------------
function createModel() {
  model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: 100,
      returnSequences: true,
      inputShape: [sequenceLength, vocab.length],
    })
  );

  model.add(
    tf.layers.lstm({
      units: 100,
    })
  );

  model.add(
    tf.layers.dense({
      units: vocab.length,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("Modell korrekt erstellt.");
}

// ------------------------------------------------------------
// TRAININGSKURVE
// ------------------------------------------------------------
function initTrainingChart() {
  const canvas = document.getElementById("trainingChart");

  if (!canvas) {
    console.warn("Canvas für Trainingskurve wurde nicht gefunden.");
    return;
  }

  const ctx = canvas.getContext("2d");

  if (trainingChart) {
    trainingChart.destroy();
  }

  trainingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochLabels,
      datasets: [
        {
          label: "Loss",
          data: lossValues,
          tension: 0.25,
        },
        {
          label: "Accuracy",
          data: accuracyValues,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: {
          display: true,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

// ------------------------------------------------------------
// TRAINING
// ------------------------------------------------------------
async function trainModel(X, y) {
  lossValues = [];
  accuracyValues = [];
  epochLabels = [];

  initTrainingChart();

  await model.fit(X, y, {
    epochs: 10,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const currentEpoch = epoch + 1;

        epochLabels.push(currentEpoch);
        lossValues.push(logs.loss);

        if (logs.accuracy !== undefined) {
          accuracyValues.push(logs.accuracy);
        } else if (logs.acc !== undefined) {
          accuracyValues.push(logs.acc);
        }

        if (trainingStatusDiv) {
          trainingStatusDiv.textContent = `Training läuft... Epoche ${currentEpoch}/10 | Loss: ${logs.loss.toFixed(
            4
          )}`;
        }

        if (trainingChart) {
          trainingChart.update();
        }

        await tf.nextFrame();
      },
    },
  });

  if (trainingStatusDiv) {
    trainingStatusDiv.textContent =
      "Training abgeschlossen. Modell wird ausgewertet...";
  }
}

// ------------------------------------------------------------
// VORHERSAGE
// ------------------------------------------------------------
function predictNextWord(inputText, topK = 5) {
  if (!modelReady) {
    showInputWarning(
      "Das Modell ist noch nicht bereit. Bitte warte, bis das Training abgeschlossen ist."
    );
    return [];
  }

  const words = cleanText(inputText);

  if (words.length < 1) {
    showInputWarning("Bitte gib mindestens ein vollständiges Wort ein.");
    return [];
  }

  // Letzte Wörter holen
  let lastWords = words.slice(-sequenceLength);

  // Fehlende Positionen mit <pad> auffüllen
  while (lastWords.length < sequenceLength) {
    lastWords.unshift("<pad>");
  }

  // Unbekannte Wörter erkennen
  const unknownWords = words.filter((w) => word2idx[w] === undefined);

  // Hinweis anzeigen, aber NICHT abbrechen
  if (unknownWords.length > 0) {
    statusDiv.textContent =
      `Hinweis: Das Modell kennt folgende Wörter nicht: ${unknownWords.join(
        ", "
      )}. ` + `Die Vorhersage läuft trotzdem weiter.`;
  } else {
    statusDiv.textContent = "Vorhersage erfolgreich berechnet.";
  }

  // Wörter in Zahlen umwandeln
  const seq = lastWords.map((w) => {
    // unbekanntes Wort
    if (word2idx[w] === undefined) {
      return word2idx["<unk>"];
    }

    return word2idx[w];
  });

  // Tensor erzeugen
  const input = tf.tensor3d([
    seq.map((idx) => {
      const oneHot = new Array(vocab.length).fill(0);
      oneHot[idx] = 1;
      return oneHot;
    }),
  ]);

  // Vorhersage
  const probs = tf.tidy(() => {
    const prediction = model.predict(input);
    return Array.from(prediction.dataSync());
  });

  input.dispose();

  // Beste Vorhersagen holen
  const topIndices = Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, topK);

  // Wörter zurückgeben
  return topIndices.map((idx) => ({
    word: idx2word[idx],
    probability: probs[idx],
  }));
}

// ------------------------------------------------------------
// UI: VORHERSAGEN ANZEIGEN
// ------------------------------------------------------------
function displayPredictions(predictions) {
  const predDiv = document.getElementById("predictions");
  predDiv.innerHTML = "";

  predictions.forEach((p) => {
    const btn = document.createElement("button");
    btn.textContent = `${p.word} (${(p.probability * 100).toFixed(1)}%)`;

    btn.onclick = () => {
      const textArea = document.getElementById("inputText");
      textArea.value += " " + p.word;

      const newPredictions = predictNextWord(textArea.value);
      displayPredictions(newPredictions);
    };

    predDiv.appendChild(btn);
  });
}

// ------------------------------------------------------------
// BUTTON EVENTS
// ------------------------------------------------------------
predictBtn.onclick = () => {
  const text = document.getElementById("inputText").value;
  const predictions = predictNextWord(text);
  if (predictions.length > 0) {
    displayPredictions(predictions);
  }
};

// Funktion für Temperature-Sampling (verhindert Endlosschleifen)
function sampleWithTemperature(probs, temperature = 0.6) {
  // 1. Logits aus den Wahrscheinlichkeiten rekonstruieren und durch Temperatur teilen
  let logits = probs.map((p) => Math.log(p + 1e-7) / temperature);

  // 2. Erneutes Softmax über die skalierten Logits berechnen
  const expLogits = logits.map((l) => Math.exp(l));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  const scaledProbs = expLogits.map((e) => e / sumExpLogits);

  // 3. Zufällige Auswahl basierend auf der neuen Wahrscheinlichkeitsverteilung
  const r = Math.random();
  let cumulativeProbability = 0;
  for (let i = 0; i < scaledProbs.length; i++) {
    cumulativeProbability += scaledProbs[i];
    if (r <= cumulativeProbability) {
      return i; // Gibt den Index des gewählten Wortes zurück
    }
  }
  return probs.indexOf(Math.max(...probs)); // Fallback: Sicherung bei Rundungsfehlern
}

// ==========================================
// WEITER-BUTTON: Nimmt strikt das k=1 Wort!
// ==========================================
nextBtn.onclick = () => {
  const textArea = document.getElementById("inputText");
  if (!modelReady || cleanText(textArea.value).length < 1) return;

  // 1. Hole die Vorhersage für den aktuellen Text
  const words = cleanText(textArea.value);
  let lastWords = words.slice(-sequenceLength);
  while (lastWords.length < sequenceLength) lastWords.unshift("<pad>");
  const seq = lastWords.map((w) =>
    word2idx[w] === undefined ? word2idx["<unk>"] : word2idx[w]
  );

  const input = tf.tensor3d([
    seq.map((idx) => {
      const oneHot = new Array(vocab.length).fill(0);
      oneHot[idx] = 1;
      return oneHot;
    }),
  ]);

  const probs = tf.tidy(() => Array.from(model.predict(input).dataSync()));
  input.dispose();

  // 2. STRIKT: Finde den Index mit der HÖCHSTEN Wahrscheinlichkeit (k=1)
  const maxIdx = probs.indexOf(Math.max(...probs));
  const chosenWord = idx2word[maxIdx];

  // 3. Text anhängen und UI updaten
  if (chosenWord && chosenWord !== "<pad>" && chosenWord !== "<unk>") {
    textArea.value = textArea.value.trim() + " " + chosenWord;
  }

  const topPredictions = predictNextWord(textArea.value);
  displayPredictions(topPredictions);
};

// ==========================================
// AUTO-BUTTON: Nutzt Temperatur NUR innerhalb der Top-5
// ==========================================
autoBtn.onclick = () => {
  let count = 0;
  const maxWords = 10;

  clearInterval(autoInterval);

  autoInterval = setInterval(() => {
    if (count >= maxWords) {
      clearInterval(autoInterval);
      statusDiv.textContent = "Automatische Vorhersage abgeschlossen.";
      return;
    }

    const textArea = document.getElementById("inputText");
    const words = cleanText(textArea.value);
    if (words.length < 1) {
      clearInterval(autoInterval);
      return;
    }

    // Vorhersage holen
    let lastWords = words.slice(-sequenceLength);
    while (lastWords.length < sequenceLength) lastWords.unshift("<pad>");
    const seq = lastWords.map((w) =>
      word2idx[w] === undefined ? word2idx["<unk>"] : word2idx[w]
    );

    const input = tf.tensor3d([
      seq.map((idx) => {
        const oneHot = new Array(vocab.length).fill(0);
        oneHot[idx] = 1;
        return oneHot;
      }),
    ]);

    const probs = tf.tidy(() => Array.from(model.predict(input).dataSync()));
    input.dispose();

    // 1. Beschränke das Sampling NUR auf die Top-5 angezeigten Wörter!
    const top5Indices = Array.from(probs.keys())
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, 5);

    const top5Probs = top5Indices.map((idx) => probs[idx]);

    // 2. Wende das Temperature-Sampling nur auf diese 5 Werte an
    const sampledTop5Idx = sampleWithTemperature(top5Probs, 0.5);
    const finalWordIdx = top5Indices[sampledTop5Idx]; // Mappe zurück auf den echten Vokabular-Index

    const chosenWord = idx2word[finalWordIdx];

    if (chosenWord && chosenWord !== "<pad>" && chosenWord !== "<unk>") {
      textArea.value = textArea.value.trim() + " " + chosenWord;
    }

    const topPredictions = predictNextWord(textArea.value);
    displayPredictions(topPredictions);

    count++;
  }, 500);
};

stopBtn.onclick = () => {
  clearInterval(autoInterval);
  statusDiv.textContent = "Automatische Vorhersage wurde unterbrochen.";
};

resetBtn.onclick = () => {
  clearInterval(autoInterval);
  document.getElementById("inputText").value = "";
  clearPredictions();
  statusDiv.textContent =
    "Eingabe zurückgesetzt. Das trainierte Modell bleibt bereit.";
};

// ------------------------------------------------------------
// EVALUATION
// ------------------------------------------------------------
function computeTopKAccuracy(X, y, kValues = [1, 5, 10, 20, 100]) {
  const topKCounts = kValues.map(() => 0);
  const total = X.shape[0];

  for (let i = 0; i < total; i++) {
    const input = X.slice([i, 0, 0], [1, X.shape[1], X.shape[2]]);
    const trueTensor = y.slice([i, 0], [1, y.shape[1]]).argMax(-1);
    const trueIdx = trueTensor.dataSync()[0];

    const preds = model.predict(input).dataSync();
    const topIndices = Array.from(preds.keys()).sort(
      (a, b) => preds[b] - preds[a]
    );

    kValues.forEach((k, idx) => {
      if (topIndices.slice(0, k).includes(trueIdx)) {
        topKCounts[idx]++;
      }
    });

    input.dispose();
    trueTensor.dispose();
  }

  const accuracies = topKCounts.map((count) => count / total);
  resultsDiv.innerHTML = "";

  kValues.forEach((k, idx) => {
    const line = document.createElement("div");
    line.className = "metric-line";
    line.textContent = `Top-${k} Accuracy: ${(accuracies[idx] * 100).toFixed(
      2
    )}%`;
    resultsDiv.appendChild(line);
  });
}

function computePerplexity(X, y) {
  const total = X.shape[0];
  let lossSum = 0;

  for (let i = 0; i < total; i++) {
    const input = X.slice([i, 0, 0], [1, X.shape[1], X.shape[2]]);
    const trueTensor = y.slice([i, 0], [1, y.shape[1]]).argMax(-1);
    const trueIdx = trueTensor.dataSync()[0];

    const preds = model.predict(input).dataSync();
    const prob = preds[trueIdx];

    lossSum += -Math.log(prob + 1e-7);

    input.dispose();
    trueTensor.dispose();
  }

  const perplexity = Math.exp(lossSum / total);

  const line = document.createElement("div");
  line.className = "metric-line";
  line.textContent = `Perplexity: ${perplexity.toFixed(3)}`;
  resultsDiv.appendChild(line);
}

// ------------------------------------------------------------
// TRAINING PIPELINE
// ------------------------------------------------------------
async function runTraining() {
  try {
    setButtonsEnabled(false);

    statusDiv.textContent = "Daten werden geladen...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent = "Daten werden geladen...";
    }

    await loadData();

    statusDiv.textContent = "Trainingsdaten werden vorbereitet...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent = "Trainingsdaten werden vorbereitet...";
    }

    const { X_tensor, y_tensor } = prepareTrainingData();

    statusDiv.textContent = "Modell wird erstellt...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent = "Modell wird erstellt...";
    }

    createModel();

    statusDiv.textContent = "Modell wird trainiert...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent =
        "Modell wird trainiert. Die Kurve wird während des Trainings aktualisiert.";
    }

    await trainModel(X_tensor, y_tensor);

    statusDiv.textContent = "Modell wird ausgewertet...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent = "Modell wird ausgewertet...";
    }

    computeTopKAccuracy(X_tensor, y_tensor);
    computePerplexity(X_tensor, y_tensor);

    statusDiv.textContent =
      "Modell bereit. Gib mindestens ein vollständiges Wort ein.";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent =
        "Modell bereit. Training und Auswertung wurden abgeschlossen.";
    }

    modelReady = true;
    setButtonsEnabled(true);

    X_tensor.dispose();
    y_tensor.dispose();
  } catch (error) {
    console.error(error);
    setButtonsEnabled(false);
    statusDiv.textContent =
      "Die Anwendung konnte nicht vollständig gestartet werden.";
  }
}

// ------------------------------------------------------------
// DROPDOWN-LOGIK
// ------------------------------------------------------------
document.querySelectorAll(".dropdown-toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const parent = btn.parentElement;
    const isOpen = parent.classList.contains("open");

    document.querySelectorAll(".dropdown").forEach((dropdown) => {
      dropdown.classList.remove("open");
    });

    if (!isOpen) {
      parent.classList.add("open");
    }
  });
});

// ------------------------------------------------------------
// START
// ------------------------------------------------------------
runTraining();
