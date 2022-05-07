using Neural_Network;

public static class ImageTest
{
    public static void Start()
    {
        Console.WriteLine("Starting Image Test!");
        string folderPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Training Data");

        (MNIST.ImageData[] trainingData, MNIST.ImageData[] testData) = MNIST.Dataset.LoadDataset(folderPath);
        var trainingSamples = trainingData
            .Select(t => new TrainingSample { Input = t.PixelsToDoubleArray(), ExpectedOutput = t.LabelToDoubleArray() })
            .ToArray();
        var testingSamples = testData
            .Select(t => new TrainingSample { Input = t.PixelsToDoubleArray(), ExpectedOutput = t.LabelToDoubleArray() })
            .ToArray();

        Console.WriteLine("Training network (This can take a while...)");
        NeuralNetwork neuralNetwork = new NeuralNetwork(28 * 28, new LayerData[] {
            new LayerData { NeuronCount = 100, ActivationFunction = ActivationFunction.ReLU },
            new LayerData { NeuronCount = 10, ActivationFunction = ActivationFunction.ReLU }
        });
        NeuralNetworkTrainer trainer = neuralNetwork.GetTrainer();
        trainer.Train(trainingSamples);
        Console.WriteLine("Testing network");
        var accuracy = trainer.Test(testingSamples, Evaluate);
        Console.WriteLine($"Accuracy: {accuracy:0.000}");

        Console.WriteLine("Serializing & deserializing network");
        var serializedNetwork = NeuralNetworkSerializer.SerializeAsBytes(neuralNetwork);
        NeuralNetwork deserializedNetwork = NeuralNetworkSerializer.DeserializeFromBytes(serializedNetwork);
        NeuralNetworkTrainer deserializedTrainer = deserializedNetwork.GetTrainer();
        Console.WriteLine("Testing deserialized network");
        var deserializedAccuracy = deserializedTrainer.Test(testingSamples, Evaluate);
        Console.WriteLine($"Accuracy: {deserializedAccuracy:0.000}");

        string serializedPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Models", "number_recognition_model.model");
        Directory.CreateDirectory(Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Models"));
        File.WriteAllBytes(serializedPath, serializedNetwork);

        //TrainAndTest(new LayerData[] {
        //    new LayerData { NeuronCount = 100, ActivationFunction = ActivationFunctionEnum.Relu },
        //    new LayerData { NeuronCount = 10, ActivationFunction = ActivationFunctionEnum.Relu }
        //}, trainingSamples, testingSamples, 10, 0.0001);

        Console.WriteLine("Done!");
        Console.WriteLine();
    }

    private static int accuracyCounter = 1;

    private static void TrainAndTest(LayerData[] layers, TrainingSample[] trainingSamples, TrainingSample[] testingSamples, int epochs = 10, double learningRate = 0.0001)
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork(28 * 28, layers);
        NeuralNetworkTrainer trainer = neuralNetwork.GetTrainer(epochs: epochs, learningRate: learningRate);
        trainer.Train(trainingSamples);
        var accuracy = trainer.Test(testingSamples, Evaluate);
        Console.WriteLine($"Accuracy {accuracyCounter++}: {accuracy:0.000}");
    }

    private static bool Evaluate(double[] output, double[] expectedOutput)
    {
        int result = GetMaxIndex(output);
        int expectedResult = GetMaxIndex(expectedOutput);
        return result == expectedResult;
    }

    private static int GetMaxIndex(double[] array)
    {
        double maxValue = array.Max();
        int maxIndex = Array.IndexOf(array, maxValue);
        return maxIndex;
    }
}