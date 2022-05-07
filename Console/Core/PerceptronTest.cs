using Neural_Network;

public static class PerceptronTest
{
    public static void Start()
    {
        const int SAMPLE_COUNT = 10000;
        const double LEARNING_RATE = 0.01;

        Console.WriteLine("Starting Perceptron Test!");
        var data = CreateTrainingSamples(SAMPLE_COUNT);

        // Split the data into training and testing sets
        int trainingCount = SAMPLE_COUNT * 8 / 10;
        var trainingSet = data.Take(trainingCount).ToArray();
        var testingSet = data.Skip(trainingCount).ToArray();

        LayerData[] layers = {
            new LayerData { NeuronCount = 2, ActivationFunction = ActivationFunction.ReLU },
            new LayerData { NeuronCount = 1, ActivationFunction = ActivationFunction.ReLU }
        };
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, layers);
        NeuralNetworkTrainer trainer = neuralNetwork.GetTrainer(learningRate: LEARNING_RATE);
        trainer.Train(trainingSet);

        var accuracy = trainer.Test(testingSet, (output, expectedOutput) => Math.Round(output[0]) == expectedOutput[0]);
        Console.WriteLine($"Accuracy: {accuracy:0.000}");
        for (int x = 0; x < 2; x++)
        {
            for (int y = 0; y < 2; y++)
            {
                var result = neuralNetwork.Fire(new double[] { x, y })[0];
                Console.WriteLine($"Inputs: ({x}, {y}): Output: ({result:0.000}, {(result > 0.5d ? 1 : 0)})");
            }
        }
        Console.WriteLine("Done!");
        Console.WriteLine();
    }

    public static TrainingSample[] CreateTrainingSamples(int sampleCount)
    {
        var samples = new TrainingSample[sampleCount];
        for (int n = 0; n < samples.Length; n++)
        {
            int input1 = RandomHelper.Next(2);
            int input2 = RandomHelper.Next(2);
            double[] inputs = { input1, input2 };
            double[] expectedOutput = { input1 == 1 && input2 == 1 ? 0 : 1 };
            samples[n] = new TrainingSample
            {
                Input = inputs,
                ExpectedOutput = expectedOutput
            };
        }

        return samples;
    }
}