# Simple-Neural-Network

A handcrafted multilayer perceptron neural network made in pure C#.

You can try it out in your browser [here](https://duzychri.github.io/projects/image-classifier).

## How to use

### Creating, Training and Testing

```csharp
NeuralNetwork neuralNetwork = new NeuralNetwork(28 * 28, new LayerData[] {
    new LayerData { NeuronCount = 100, ActivationFunction = ActivationFunction.ReLU },
    new LayerData { NeuronCount = 10, ActivationFunction = ActivationFunction.ReLU }
});
TrainingSample[] trainingSamples = CreateTrainingSamples();
NeuralNetworkTrainer trainer = neuralNetwork.GetTrainer(epochs, learningRate);
trainer.Train(trainingSamples);
double accuracy = trainer.Test(testingSamples, Evaluate);
```

### Serializing & Deserializing

Serializing uses protobuf-net to serialize to binary and then a gzip stream for compression.

```csharp
byte[] serializedNetwork = NeuralNetworkSerializer.SerializeAsBytes(neuralNetwork);
NeuralNetwork deserializedNetwork = NeuralNetworkSerializer.DeserializeFromBytes(serializedNetwork);
```

### Using

```csharp
Image image = Image.FromFile("...");
double[] pixels = ExtractPixelsFromImage(image);
double[] outputValues = network.Fire(pixels);
```