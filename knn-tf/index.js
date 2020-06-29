require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k) {
  // We need to normalize the data due that sqft_lot is very spread (irregularly) and lat and long are very similar
  // we will use standard deviation instead of minmax because the sqft_lot is very iregular field, a few very very big and the rest smaller.
  // the formula is value - average / sqrt(variance)  
  // tensorflow does the hard part of geting the variance and average using moments()
  const { mean, variance } = tf.moments(features, 0); //0 is the axis of the operation
  const scaledPreduction = predictionPoint.sub(mean).div(variance.pow(.5)); //we also apply the standard deviation formula to the predictionPoint

  return features
    .sub(mean).div(variance.pow(.5)) //we apply the standard deviation to the features
    .sub(scaledPreduction)
    .pow(2) // till here we have a tensor shaped [[n, m]] (where n is the number of rows in features tensor and m is the number of columns)
    .sum(1) // will sum and group results in a tensor shaped [[n]]
    .pow(.5)
    .expandDims(1) //we need to expand the dimension loose with sum the then use concat.
    .concat(labels, 1)
    .unstack()  // we wnat to sort the rows by distance but tensorflow does not allow to sort values inside a tensor.
                // with unstack we get a normal JS array where each element is a tenson shaped [2] (n elements) 
                // instead of having one tensor with shape [n, 2]
    .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1) //now that we have a normal JS array we use native JS array.sort
    .slice(0, k) // this is still a JS array so this slice method is from JS array not from tensorflow (tf.slice)
    .reduce((acc, pair) => acc + pair.get(1), 0) / k;
}

let { features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
  labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) * 100 / testLabels[i][0] //percentage of error
  console.log('Guess', err);  
});



