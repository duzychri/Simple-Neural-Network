using Neural_Network;
using System.Drawing;

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

        //foreach (var t in testData)
        //{
        //    string folderPath2 = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Resources2");
        //    Directory.CreateDirectory(folderPath2);
        //    t.ToImage().Save(Path.Combine(folderPath2, t.label + ".png"));
        //}

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

        //TestWithOtherImages(deserializedNetwork);

        string serializedPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Models", "number_recognition_model.model");
        Directory.CreateDirectory(Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Models"));
        File.WriteAllBytes(serializedPath, serializedNetwork);

        Console.WriteLine("Done!");
        Console.WriteLine();
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

    private static void TestWithOtherImages(NeuralNetwork network)
    {
        Console.WriteLine("Testing deserialized network with custom images");
        string folderPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Resources");

        var fileNames = Directory.GetFiles(folderPath).Select(f => Path.GetFileName(f).Split('.')[0]).Distinct();

        foreach (string fileName in fileNames)
        {
            int expectedOutput = int.Parse(fileName);
            string imagePath = Path.Combine(folderPath, fileName + ".png");
            Image image = Image.FromFile(imagePath);
            double[] pixels = ExtractPixelsFromImage(image).Select(p => (double)p).ToArray();
            double[] outputValues = network.Fire(pixels);
            int output = Array.IndexOf(outputValues, outputValues.Max());
            //Console.WriteLine($"expected: {expectedOutput}, output: {output}, success: {expectedOutput == output}");
        }
    }

    private static byte[] ExtractPixelsFromImage(Image image)
    {
        Bitmap bitmap = image as Bitmap ?? new Bitmap(image);
        byte[] pixels = new byte[bitmap.Width * bitmap.Height * 1];
        int pixelsIndex = 0;
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                Color pixel = bitmap.GetPixel(x, y);
                pixels[pixelsIndex++] = pixel.R;
                //pixels[pixelsIndex++] = pixel.G;
                //pixels[pixelsIndex++] = pixel.B;
                //pixels[pixelsIndex++] = pixel.A;
            }
        }
        return pixels;
    }
}