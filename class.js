
var model;

async function initialize(callback) {
    model = await tf.loadLayersModel('models/model.json');
    callback();
}
window.predict = async(image) => {
    // action for the submit button
    let tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([64, 64]).toFloat().expandDims();
    prediction = await model.predict(tensorImg).data();
    return prediction;
}
