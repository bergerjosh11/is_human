const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const app = express();
const PORT = process.env.PORT || 3000;

const upload = multer({ dest: 'uploads/' });

// Load the model
let model;
cocoSsd.load()
  .then(loadedModel => {
    model = loadedModel;
    console.log('Model loaded');
  })
  .catch(err => {
    console.error('Error loading model:', err);
  });

// Middleware to serve static files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Endpoint to upload image and check if it contains a person
app.post('/upload', upload.single('image'), async (req, res) => {
  if (!model) {
    return res.status(500).json({ error: 'Model not loaded' });
  }

  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const imgPath = path.join(__dirname, req.file.path);

  try {
    // Resize the image to 300x300
    const resizedImageBuffer = await sharp(imgPath).resize(300, 300).toBuffer();

    const imgTensor = tf.node.decodeImage(resizedImageBuffer);

    const predictions = await model.detect(imgTensor);

    const containsPerson = predictions.some(prediction => prediction.class === 'person');

    // Clean up
    fs.unlinkSync(imgPath);

    res.json({ containsPerson });
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ error: 'Failed to process image' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
