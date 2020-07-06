require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['passedemissions'],
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: (value) => value === 'TRUE' ? 1 : 0
  }
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5, //does not matter because the class automatically improves it
  iterations: 100,
  batchSize: 10,
  //desicionBoundary: 0.6
});

regression.train();

console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse()
});
// regression.predict([
//   // ['horsepower', 'weight', 'displacement']
//   [88, 97, 1.065]

// ]).print(); //some issue in lesson 124