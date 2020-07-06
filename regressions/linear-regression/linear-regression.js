const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {

  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 }, 
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
      
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /**
   * Gradient descent with tensorflow. See lesson  84, 85, 86
   */
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights); //matmul is matrix multiplication. mul() is just and ordinary multiplication of a tensor and a discrete value
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
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);
    
    const res = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .get(); //with get we get the output of the operation outside the tensor to have a plain value

    const tot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    return 1 - res / tot;
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
   * Calculate MSE mean square error and store it in an array in reverse order
   */
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get()

    this.mseHistory.unshift(mse); //unshift is th oposite to push, it insert the element in the front of the array
  }

  /**
   * Update the learning curve automatically  based in the progression of the MSE to avoid manually testing different values 
   */
  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;