/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);
  return cleaned;
}

function createModel() {
  // Create a sequential model
  // Sequencial flui da entrada para a saída.
  const model = tf.sequential();

  // Add a single input layer
  // A camada de entrada é automaticamente conectada a uma camada oculta.
  // Uma camada densa multiplica os inputs por uma matriz. Nesse caso a matriz é 1x1, ou seja
  // o input será multiplicado por um número. Units é 1 pois haverá um peso para cada entrada.
  // Bias é o que será adicionado após a multiplicação. Tem default para true, então poderia ser omitido.
  // inputShape corresponde ao número de entradas da rede.
  model.add(tf.layers.dense({ inputShape: [1], units: 50, useBias: true }));
  // Uma função de ativação é aplicada entre uma camada e outra.
  // https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    // Normalizar depois de transformar em tensor ajuda já que podemos pegar o valor max e min com facilidade.
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    // Esse código subtrai (.sub) cada elemento do tensor de um valor, e depois divide (.div) por outros valores
    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    // O otimizador Adam não precisa de configuração.
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    // Cria 100 exemplos com ranges uniformes entre 0 e 1.
    const xs = tf.linspace(0, 1, 100);
    // O reshape serve para colocar os exemplos no mesmo formato [num_examples, num_features_per_example] do tensor
    const preds = model.predict(xs.reshape([100, 1]));

    // Desnormalização dos exemplos e das predições.
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // O método dataSync traz os dados de volta para um array JS simples. Antes era um tensor.
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  // Renders on body
  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

document.addEventListener("DOMContentLoaded", run);
