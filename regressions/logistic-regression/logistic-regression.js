const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {

  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = []; // metric of how bad we guessed

    
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, desicionBoundary: 0.5 }, 
      options
    );

    this.weights = tf.zeros([this.features.shape[1],1]); //a tensor with n (the amount of features columns (that represents diff car caracteristics)) rows and 1 columns of zeors ( similar to above tf.ones())
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
    const { batchSize } = this.options;

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice(
          [ startIndex, 0 ], //the row & column to start the slice
          [ batchSize, -1 ] //the amount of rows & columns in the slice (-1 means all columns in the row)
        );

        const labelSlice = this.labels.slice(
          [ startIndex, 0 ], //the row & column to start the slice
          [ batchSize, -1 ] //the amount of rows & columns in the slice (-1 means all columns in the row)
        );

        this.gradientDescent(featureSlice, labelSlice);
      }
      
      this.recordCost();
      this.updateLearningRate();
    }
  }

  /**
   * Gradient descent with tensorflow. See lesson  84, 85, 86
   */
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid(); //matmul is matrix multiplication. mul() is just and ordinary multiplication of a tensor and a discrete value
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]); //number of rows in features (car records)

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   * Gradient descent using arrays and mathematics "manually". This method was used to understand the maths of gradient descent. 
   * IGNORE IT
   */
  gradientDescentWithMath() {
    //see lesson 77 for this equations
    const currentGuessesForMPG = this.features.map(row => this.m * row[0] + this.b);
    
    const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => { 
      return guess - this.labels[i][0];
    })) * 2 / this.features.length;

    const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => { 
      return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    })) * 2 / this.features.length;

    this.b = this.b - bSlope * this.options.learningRate;
    this.m = this.m - mSlope * this.options.learningRate;
  }

  // predict a mpg value based on provided observations
  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.desicionBoundary) //this converts all the values in boolean 
      .cast('float32'); //we need to turn it numbers again to use it for the following math operations
  }

  test(testFeatures, testLabels) {
    //see lesson 125 on how to test the data
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /**
   * returns a normalized features vector with also a contatenations of a ones column
   */
  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standarize(features); //get the standard deviaition
    features = tf.ones([features.shape[0], 1]).concat(features, 1); //add the column of ones

    return features;
  }

  /**
   * calculate standard deviation
   * @param {*} features 
   */
  standarize(features) {
    if (!this.mean) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      this.variance = variance;
    }

    return features.sub(this.mean).div(this.variance.pow(0.5)); //standard deviation
  }

  /**
   * Calculate Cost and store it in an array in reverse order
   * See lesson 128, 129 for this formula
   */
  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne.add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);
    
    this.costHistory.unshift(cost);
  }

  /**
   * Update the learning curve automatically  based in the progression of the cost to avoid manually testing different values 
   */
  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;