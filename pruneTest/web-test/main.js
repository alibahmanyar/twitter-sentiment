// Unusable due to https://github.com/tensorflow/tfjs/issues/6242
async function testModel() {
    const tfliteModel = await tflite.loadTFLiteModel('/model.tflite');
}
testModel();