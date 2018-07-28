# sketch_rnn_pytorch
Implementation of RNN-VAE following the work of Sketch-RNN using pytorch.
https://magenta.tensorflow.org/sketch_rnn

## Implementation details
The loss consists three parts.
1) Mean square error of the decoder network
2) Cross entropy error of the stroke type
3) KL-divergence loss

## Dataset: 
https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn/?pli=1
For cat:
https://storage.cloud.google.com/quickdraw_dataset/sketchrnn/cat.npz


