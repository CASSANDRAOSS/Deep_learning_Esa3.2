let vocab = [];
let word2idx = {};
let idx2word = {};
let sequences = [];
let nextWords = [];
const sequenceLength = 5;

let model;
let modelReady = false;

// Buttons & UI-Elemente
const predictBtn = document.getElementById("predictBtn");
const nextBtn = document.getElementById("nextBtn");
const autoBtn = document.getElementById("autoBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");

function setButtonsEnabled(enabled) {
    [predictBtn, nextBtn, autoBtn, stopBtn, resetBtn].forEach(btn => {
        btn.disabled = !enabled;
    });
}

// ------------------------------------------------------------
// DATEN LADEN
// ------------------------------------------------------------
async function loadData() {
    const response = await fetch('data/text_corpus.txt');
    const text = await response.text();

    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);

    vocab = Array.from(new Set(words));
    vocab.forEach((word, idx) => {
        word2idx[word] = idx;
        idx2word[idx] = word;
    });

    for (let i = 0; i <= words.length - sequenceLength - 1; i++) {
        const seq = words.slice(i, i + sequenceLength);
        sequences.push(seq.map(w => word2idx[w]));
        nextWords.push(word2idx[words[i + sequenceLength]]);
    }

    console.log(`Vokabulargröße: ${vocab.length}`);
    console.log(`Anzahl der Sequenzen: ${sequences.length}`);
}

// ------------------------------------------------------------
// TRAININGSDATEN VORBEREITEN
// ------------------------------------------------------------
function prepareTrainingData() {
    const X = [];
    const y = [];

    sequences.forEach((seq, i) => {
        const xSeq = seq.map(idx => {
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
        y_tensor: tf.tensor2d(y)
    };
}

// ------------------------------------------------------------
// MODELL ERSTELLEN
// ------------------------------------------------------------
function createModel() {
    model = tf.sequential();

    model.add(tf.layers.lstm({
        units: 100,
        returnSequences: true,
        inputShape: [sequenceLength, vocab.length]
    }));

    model.add(tf.layers.lstm({ units: 100 }));

    model.add(tf.layers.dense({
        units: vocab.length,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy'
    });

    console.log("Modell erstellt:");
    model.summary();
}

// ------------------------------------------------------------
// TRAINING
// ------------------------------------------------------------
async function trainModel(X, y) {
    await model.fit(X, y, {
        epochs: 50,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss=${logs.loss.toFixed(4)}`);
            }
        }
    });
}

// ------------------------------------------------------------
// VORHERSAGE
// ------------------------------------------------------------
function predictNextWord(inputText, topK = 5) {
    if (!modelReady) return [];

    const words = inputText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    if (words.length < sequenceLength) return [];

    const seq = words.slice(-sequenceLength).map(w => word2idx[w] || 0);

    const input = tf.tensor3d([seq.map(idx => {
        const oneHot = new Array(vocab.length).fill(0);
        oneHot[idx] = 1;
        return oneHot;
    })]);

    const prediction = model.predict(input);
    const probs = prediction.dataSync();

    const topIndices = Array.from(probs.keys())
        .sort((a, b) => probs[b] - probs[a])
        .slice(0, topK);

    return topIndices.map(idx => ({
        word: idx2word[idx],
        probability: probs[idx]
    }));
}

// ------------------------------------------------------------
// UI: VORHERSAGEN ANZEIGEN
// ------------------------------------------------------------
function displayPredictions(predictions) {
    const predDiv = document.getElementById("predictions");
    predDiv.innerHTML = '';

    predictions.forEach(p => {
        const btn = document.createElement('button');
        btn.textContent = `${p.word} (${(p.probability * 100).toFixed(1)}%)`;
        btn.onclick = () => {
            const textArea = document.getElementById("inputText");
            textArea.value += ' ' + p.word;

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
    displayPredictions(predictions);
};

nextBtn.onclick = () => {
    const textArea = document.getElementById("inputText");
    let text = textArea.value;

    const predictions = predictNextWord(text, 1);
    if (predictions.length === 0) return;

    const nextWord = predictions[0].word;
    textArea.value = text + ' ' + nextWord;

    const topPredictions = predictNextWord(textArea.value);
    displayPredictions(topPredictions);
};

resetBtn.onclick = () => {
    document.getElementById("inputText").value = '';
    document.getElementById("predictions").innerHTML = '';
};

let autoInterval;
autoBtn.onclick = () => {
    let count = 0;
    const maxWords = 10;

    autoInterval = setInterval(() => {
        if (count >= maxWords) {
            clearInterval(autoInterval);
            return;
        }

        const textArea = document.getElementById("inputText");
        const predictions = predictNextWord(textArea.value, 1);
        if (predictions.length === 0) return;

        textArea.value += ' ' + predictions[0].word;

        const topPredictions = predictNextWord(textArea.value);
        displayPredictions(topPredictions);

        count++;
    }, 500);
};

stopBtn.onclick = () => {
    clearInterval(autoInterval);
};

// ------------------------------------------------------------
// EVALUATION
// ------------------------------------------------------------
function computeTopKAccuracy(X, y, kValues = [1, 5, 10, 20, 100]) {
    const topKCounts = kValues.map(_ => 0);
    const total = X.shape[0];

    for (let i = 0; i < total; i++) {
        const input = X.slice([i, 0, 0], [1, X.shape[1], X.shape[2]]);
        const trueIdx = y.slice([i, 0], [1, y.shape[1]]).argMax(-1).dataSync()[0];

        const preds = model.predict(input).dataSync();
        const topIndices = Array.from(preds.keys())
            .sort((a, b) => preds[b] - preds[a]);

        kValues.forEach((k, idx) => {
            if (topIndices.slice(0, k).includes(trueIdx)) topKCounts[idx]++;
        });
    }

    const accuracies = topKCounts.map(count => count / total);
    resultsDiv.innerHTML = "";

    kValues.forEach((k, idx) => {
        const line = document.createElement('div');
        line.textContent = `Top-${k} Accuracy: ${(accuracies[idx] * 100).toFixed(2)}%`;
        resultsDiv.appendChild(line);
    });
}

function computePerplexity(X, y) {
    const total = X.shape[0];
    let lossSum = 0;

    for (let i = 0; i < total; i++) {
        const input = X.slice([i, 0, 0], [1, X.shape[1], X.shape[2]]);
        const trueIdx = y.slice([i, 0], [1, y.shape[1]]).argMax(-1).dataSync()[0];

        const preds = model.predict(input).dataSync();
        const prob = preds[trueIdx];
        lossSum += -Math.log(prob + 1e-7);
    }

    const perplexity = Math.exp(lossSum / total);
    const line = document.createElement('div');
    line.textContent = `Perplexity: ${perplexity.toFixed(3)}`;
    resultsDiv.appendChild(line);
}

// ------------------------------------------------------------
// TRAINING PIPELINE
// ------------------------------------------------------------
async function runTraining() {
    setButtonsEnabled(false);
    statusDiv.textContent = "Daten werden geladen...";

    await loadData();

    statusDiv.textContent = "Trainingsdaten werden vorbereitet...";
    const { X_tensor, y_tensor } = prepareTrainingData();

    statusDiv.textContent = "Modell wird erstellt...";
    createModel();

    statusDiv.textContent = "Modell wird trainiert...";
    await trainModel(X_tensor, y_tensor);

    statusDiv.textContent = "Modell wird ausgewertet...";
    computeTopKAccuracy(X_tensor, y_tensor);
    computePerplexity(X_tensor, y_tensor);

    statusDiv.textContent = "Modell bereit. Gib einen Text ein.";
    modelReady = true;
    setButtonsEnabled(true);
}

// Dropdown-Logik
document.querySelectorAll(".dropdown-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
        const parent = btn.parentElement;
        parent.classList.toggle("open");
    });
});


// ------------------------------------------------------------
// START
// ------------------------------------------------------------
runTraining();
