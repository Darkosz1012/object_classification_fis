
var model;
async function initialize() {
    model = await tf.loadLayersModel('models/model.json');
    model.summary();
    console.log(model)
}
window.predict = async(image) => {
    // action for the submit button
    console.log(tf.browser.fromPixels(image))
    let tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([180, 180]).toFloat().expandDims();
    prediction = await model.predict(tensorImg).data();
    return prediction;
}
initialize();