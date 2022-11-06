using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MlNet;

public class MlNetImageData
{
    public float Label { get; set; }

    [VectorType(784)]
    public float[] Features { get; set; }
}

public class MlNetOutputData
{
    [ColumnName("Score")]
    public float[] Score;
}

public static class MlNetTest
{
    private static readonly string modelSavePath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Models", "ml_net_model.zip");

    public static string GetAbsolutePath(string relativePath)
    {
        FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
        string assemblyFolderPath = _dataRoot.Directory.FullName;

        string fullPath = Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    public static void Start()
    {
        Console.WriteLine("Starting ML .Net Test!");

        Console.WriteLine("Load training data");
        (MNIST.ImageData[] trainingData, MNIST.ImageData[] testData) = MNIST.Dataset.LoadDataset();
        var trainingSamples = trainingData
            .Select(t => new MlNetImageData { Features = t.PixelsToFloatArray(), Label = t.label })
            .ToArray();
        var testingSamples = testData
            .Select(t => new MlNetImageData { Features = t.PixelsToFloatArray(), Label = t.label })
            .ToArray();

        Console.WriteLine("Create ML context");
        MLContext context = new MLContext();

        Console.WriteLine("Create training data");
        IDataView trainingDataView = context.Data.LoadFromEnumerable(trainingSamples);
        IDataView testingDataView = context.Data.LoadFromEnumerable(testingSamples);

        Console.WriteLine("Set training algorithm");
        var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: nameof(MlNetImageData.Label), featureColumnName: nameof(MlNetImageData.Features));

        Console.WriteLine("Configure pipeline");
        var trainingPipeline = context.Transforms.Conversion
            .MapValueToKey("Label", nameof(MlNetImageData.Label), keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
            .Append(context.Transforms
                .Concatenate("Features", nameof(MlNetImageData.Features))
                .AppendCacheCheckpoint(context)
            )
            .Append(trainer);
        //.Append(mlContext.Transforms.Conversion.MapKeyToValue(nameof(MlNetImageData.Label), "Label"));

        Console.WriteLine("Training network (This can take a while...)");
        ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

        Console.WriteLine("Testing deserialized network");
        IDataView predictions = trainedModel.Transform(testingDataView);
        MulticlassClassificationMetrics metrics = context.MulticlassClassification.Evaluate(data: predictions, labelColumnName: nameof(MlNetImageData.Label), scoreColumnName: "Score");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:0.000}, MacroAccuracy: {metrics.MacroAccuracy:0.000}");

        context.Model.Save(trainedModel, trainingDataView.Schema, modelSavePath);
    }

    public static float[] TestForOne()
    {
        (MNIST.ImageData[] _, MNIST.ImageData[] testData) = MNIST.Dataset.LoadDataset();
        var testingSamples = testData
            .Select(t => new MlNetImageData { Features = t.PixelsToFloatArray(), Label = t.label })
            .ToArray();

        MLContext context = new MLContext();
        ITransformer trainedModel = context.Model.Load(modelSavePath, out DataViewSchema _);
        PredictionEngine<MlNetImageData, MlNetOutputData> predictionEngine = context.Model.CreatePredictionEngine<MlNetImageData, MlNetOutputData>(trainedModel);
        MlNetOutputData result = predictionEngine.Predict(testingSamples[0]);
        Console.WriteLine(GetMaxIndex(result.Score));
        return result.Score;
    }

    private static int GetMaxIndex(float[] array)
    {
        float maxValue = array.Max();
        int maxIndex = Array.IndexOf(array, maxValue);
        return maxIndex;
    }
}