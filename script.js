let vocab = []; // Hier landen alle einzelnen Wörter aus dem Text
let word2idx = {}; // Zum Nachschlagen: Welches Wort hat welche Nummer?
let idx2word = {}; // Zum Nachschlagen: Welche Nummer gehört zu welchem Wort?
let sequences = []; // Die Text-Häppchen fürs Training (X)
let nextWords = []; // Das jeweils darauffolgende richtige Wort (y)
const sequenceLength = 5; // Das Modell schaut sich immer 5 Wörter an

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

// Macht alle Buttons während des Trainings aus, damit man nichts durcheinanderbringt
function setButtonsEnabled(enabled) {
  [predictBtn, nextBtn, autoBtn, stopBtn, resetBtn].forEach((btn) => {
    btn.disabled = !enabled;
  });
}

// ------------------------------------------------------------
// HILFSFUNKTIONEN
// ------------------------------------------------------------
function cleanText(text) {
  // Alles klein schreiben, Satzzeichen löschen und bei Leerzeichen trennen
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

    // Einzigartige Wörter filtern und Platzhalter für Padding und unbekannte Wörter einfügen
    vocab = ["<pad>", "<unk>", ...Array.from(new Set(words))];

    // Die Übersetzungslisten (Wort zu Zahl und umgekehrt) befüllen
    vocab.forEach((word, idx) => {
      word2idx[word] = idx;
      idx2word[idx] = word;
    });

    // Schiebefenster: Geht durch den Text und packt immer 5 Wörter in X und das nächste Wort in y
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

  // Wörter in Nullen und Einsen umwandeln (One-Hot), weil das Netz keine Strings versteht
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

  // Alles in Tensoren packen, damit TensorFlow.js damit arbeiten kann
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

  // Erste LSTM-Schicht (braucht returnSequences, damit die nächste Schicht auch Sequenzen kriegt)
  model.add(
    tf.layers.lstm({
      units: 100,
      returnSequences: true,
      inputShape: [sequenceLength, vocab.length],
    })
  );

  // Zweite LSTM-Schicht für mehr Tiefe bei der Satzstruktur
  model.add(
    tf.layers.lstm({
      units: 100,
    })
  );

  // Ausgangsschicht: Softmax gibt uns Wahrscheinlichkeiten für jedes einzelne Wort aus
  model.add(
    tf.layers.dense({
      units: vocab.length,
      activation: "softmax",
    })
  );

  // Optimizer und Fehlerfunktion festlegen
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
        { label: "Loss", data: lossValues, tension: 0.25 },
        { label: "Accuracy", data: accuracyValues, tension: 0.25 },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: true } },
      scales: { y: { beginAtZero: true } },
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

  // Eigentliches Training starten (15 Durchgänge durch die Daten)
  await model.fit(X, y, {
    epochs: 15,
    batchSize: 32,
    shuffle: true, // Sätze durchmischen, damit die Reihenfolge nicht auswendig gelernt wird
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
          trainingStatusDiv.textContent = `Training läuft... Epoche ${currentEpoch}/15 | Loss: ${logs.loss.toFixed(
            4
          )}`;
        }

        if (trainingChart) {
          trainingChart.update(); // Diagramm live updaten
        }

        // Dem Browser kurz Zeit geben, um die Seite flüssig neu zu zeichnen
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

  // Die letzten geschriebenen Wörter für den Kontext rausschneiden
  let lastWords = words.slice(-sequenceLength);

  // Wenn weniger als 5 Wörter da sind, von links mit <pad> auffüllen
  while (lastWords.length < sequenceLength) {
    lastWords.unshift("<pad>");
  }

  // Schauen, ob Wörter eingegeben wurden, die gar nicht im Textkorpus vorkamen
  const unknownWords = words.filter((w) => word2idx[w] === undefined);

  if (unknownWords.length > 0) {
    statusDiv.textContent = `Hinweis: Das Modell kennt folgende Wörter nicht: ${unknownWords.join(
      ", "
    )}. Die Vorhersage läuft trotzdem weiter.`;
  } else {
    statusDiv.textContent = "Vorhersage erfolgreich berechnet.";
  }

  // Wörter in die passenden Zahlen-IDs übersetzen (unbekannte kriegen <unk>)
  const seq = lastWords.map((w) => {
    if (word2idx[w] === undefined) return word2idx["<unk>"];
    return word2idx[w];
  });

  // Tensor für die Vorhersage bauen
  const input = tf.tensor3d([
    seq.map((idx) => {
      const oneHot = new Array(vocab.length).fill(0);
      oneHot[idx] = 1;
      return oneHot;
    }),
  ]);

  // Vorhersage ausführen (tf.tidy räumt danach den Grafikspeicher auf)
  let probs = tf.tidy(() => {
    const prediction = model.predict(input);
    return Array.from(prediction.dataSync());
  });

  input.dispose();

  // ============================================================
  // ERWEITERTE SCHLEIFEN-BREMSE & STOPPWORT-REGULIERUNG
  // ============================================================

  // 1. Wiederholungssperre: Wörter aus den letzten 6 Positionen raussuchen und prozentual abstrafen
  const recentWords = words.slice(-6);
  recentWords.forEach((w) => {
    const idx = word2idx[w];
    if (idx !== undefined && w !== "<pad>" && w !== "<unk>") {
      probs[idx] *= 0.05; // Wahrscheinlichkeit massiv runterdrücken
    }
  });

  // 2. Stoppwort-Bremse: Wenn das Modell ratlos ist (Beste Option unter 15%),
  // drücken wir Füllwörter weg, damit sinnvolle Inhaltswörter nach oben rutschen
  const maxProb = Math.max(...probs);
  if (maxProb < 0.15) {
    const stopWords = [
      "und",
      "die",
      "der",
      "das",
      "ein",
      "eine",
      "ist",
      "in",
      "zu",
      "mit",
      "von",
    ];
    stopWords.forEach((word) => {
      const idx = word2idx[word];
      if (idx !== undefined) {
        probs[idx] *= 0.1;
      }
    });
  }
  // ============================================================

  // Die Ergebnisse sortieren, damit die wahrscheinlichsten Wörter ganz vorne stehen
  const topIndices = Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, topK);

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

  // Für jedes vorgeschlagene Wort einen klickbaren Button erstellen
  predictions.forEach((p) => {
    const btn = document.createElement("button");
    btn.textContent = `${p.word} (${(p.probability * 100).toFixed(1)}%)`;

    btn.onclick = () => {
      const textArea = document.getElementById("inputText");
      textArea.value += " " + p.word; // Wort im Textfeld ergänzen

      // Sofort die Vorschläge für das neue Satzende berechnen
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

// Berechnet eine etwas zufälligere Auswahl für den Auto-Button anhand der Temperatur (0.6)
function sampleWithTemperature(probs, temperature = 0.6) {
  // 1. Logits künstlich verzerren (durch die Temperatur teilen)
  let logits = probs.map((p) => Math.log(p + 1e-7) / temperature);

  // 2. Softmax neu berechnen, um wieder gültige Prozentwerte zu kriegen
  const expLogits = logits.map((l) => Math.exp(l));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  const scaledProbs = expLogits.map((e) => e / sumExpLogits);

  // 3. Basierend auf der neuen Verteilung zufällig ein Wort ziehen
  const r = Math.random();
  let cumulativeProbability = 0;
  for (let i = 0; i < scaledProbs.length; i++) {
    cumulativeProbability += scaledProbs[i];
    if (r <= cumulativeProbability) {
      return i;
    }
  }
  return probs.indexOf(Math.max(...probs)); // Fallback, falls bei den Kommastellen was schiefgeht
}

// ==========================================
// WEITER-BUTTON: Nimmt stur das wahrscheinlichste Wort (Platz 1)
// ==========================================
nextBtn.onclick = () => {
  const textArea = document.getElementById("inputText");
  if (!modelReady || cleanText(textArea.value).length < 1) return;

  const predictions = predictNextWord(textArea.value, 5);
  if (predictions.length === 0) return;

  let chosenWord = predictions[0].word; // Stur Platz 1 nehmen

  // Falls Platz 1 ein technischer Platzhalter ist, weichen wir auf Platz 2 aus
  if (chosenWord === "<pad>" || chosenWord === "<unk>") {
    chosenWord = predictions[1] ? predictions[1].word : "";
  }

  if (chosenWord) {
    textArea.value = textArea.value.trim() + " " + chosenWord;
  }

  const topPredictions = predictNextWord(textArea.value);
  displayPredictions(topPredictions);
};

// ============================================================
// AUTO-BUTTON: Schreibt von alleine bis zu 10 Wörter weiter
// ============================================================
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
    if (cleanText(textArea.value).length < 1) {
      clearInterval(autoInterval);
      return;
    }

    // 1. Hole die aktuell angezeigten Top-5 Wörter
    const predictions = predictNextWord(textArea.value, 5);
    if (predictions.length === 0) {
      clearInterval(autoInterval);
      return;
    }

    // 2. System-Tokens rausfiltern
    const validPredictions = predictions.filter(
      (p) => p.word !== "<pad>" && p.word !== "<unk>"
    );
    if (validPredictions.length === 0) {
      clearInterval(autoInterval);
      return;
    }

    // 3. Nur die Prozentwerte der im UI sichtbaren Wörter nehmen
    const probs = validPredictions.map((p) => p.probability);

    // 4. Temperatur-Zufall auf diese Auswahl anwenden
    const sampledIdx = sampleWithTemperature(probs, 0.6);
    const chosenWord = validPredictions[sampledIdx].word;

    // 5. Text im Textfeld ergänzen
    textArea.value = textArea.value.trim() + " " + chosenWord;

    // 6. UI-Vorschläge sofort für das NEUE Textende berechnen, damit Buttons und Text zusammenpassen
    const nextPredictions = predictNextWord(textArea.value, 5);
    displayPredictions(nextPredictions);

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

  // Testen, wie oft das echte nächste Wort in den Top-K Vorschlägen des Modells auftaucht
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

// Berechnet die Unsicherheit des Modells (je kleiner, desto treffsicherer ist es)
function computePerplexity(X, y) {
  const total = X.shape[0];
  let lossSum = 0;

  for (let i = 0; i < total; i++) {
    const input = X.slice([i, 0, 0], [1, X.shape[1], X.shape[2]]);
    const trueTensor = y.slice([i, 0], [1, y.shape[1]]).argMax(-1);
    const trueIdx = trueTensor.dataSync()[0];

    const preds = model.predict(input).dataSync();
    const prob = preds[trueIdx];

    lossSum += -Math.log(prob + 1e-7); // Fehlerwerte aufsummieren

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
    if (trainingStatusDiv)
      trainingStatusDiv.textContent = "Daten werden geladen...";
    await loadData();

    statusDiv.textContent = "Trainingsdaten werden vorbereitet...";
    if (trainingStatusDiv)
      trainingStatusDiv.textContent = "Trainingsdaten werden vorbereitet...";
    const { X_tensor, y_tensor } = prepareTrainingData();

    statusDiv.textContent = "Modell wird erstellt...";
    if (trainingStatusDiv)
      trainingStatusDiv.textContent = "Modell wird erstellt...";
    createModel();

    statusDiv.textContent = "Modell wird trainiert...";
    if (trainingStatusDiv) {
      trainingStatusDiv.textContent =
        "Modell wird trainiert. Die Kurve wird während des Trainings aktualisiert.";
    }
    await trainModel(X_tensor, y_tensor);

    statusDiv.textContent = "Modell wird ausgewertet...";
    if (trainingStatusDiv)
      trainingStatusDiv.textContent = "Modell wird ausgewertet...";
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
