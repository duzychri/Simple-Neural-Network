using System.Drawing;

namespace MNIST;

public class ImageData
{
    public byte label;
    public byte[,] pixels;

    public ImageData(byte label, byte[,] pixels)
    {
        this.label = label;
        this.pixels = pixels;
    }

    public float[] LabelToFloatArray()
    {
        return LabelToDoubleArray().Select(p => (float)p).ToArray();
    }

    public double[] LabelToDoubleArray()
    {
        double[] result = new double[10];
        result[label] = 1;
        return result;
    }

    public float[] PixelsToFloatArray()
    {
        return PixelsToDoubleArray().Select(p => (float)p).ToArray();
    }

    public double[] PixelsToDoubleArray()
    {
        int width = pixels.GetLength(0);
        int height = pixels.GetLength(1);
        double[] result = new double[width * height];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                result[x + width * y] = pixels[x, y];
            }
        }
        return result;
    }

    public Image ToImage()
    {
        int width = pixels.GetLength(0);
        int height = pixels.GetLength(1);
        Bitmap bitmap = new Bitmap(width, height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Color color = Color.FromArgb(255, pixels[x, y], pixels[x, y], pixels[x, y]);
                bitmap.SetPixel(x, y, color);
            }
        }
        return bitmap;
    }
}

/// <remarks>
/// http://yann.lecun.com/exdb/mnist
/// </remarks>
public static class Dataset
{
    private static ImageData[] cachedTrainingData = null;
    private static ImageData[] cachedTestData = null;

    public static (ImageData[] trainingData, ImageData[] testData) LoadDataset()
    {
        if (cachedTrainingData != null && cachedTestData != null)
        { return (cachedTrainingData, cachedTestData); }

        string folderPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "Training Data");

        string compressedFolderPath = Path.Combine(folderPath, "Compressed");
        string uncompressedFolderPath = Path.Combine(folderPath, "Uncompressed");
        Directory.CreateDirectory(uncompressedFolderPath);

        string compressedTrainImages = Path.Combine(compressedFolderPath, "train-images-idx3-ubyte.gz");
        string uncompressedTrainImages = Path.Combine(uncompressedFolderPath, "train-images.idx3-ubyte");

        string compressedTrainLabels = Path.Combine(compressedFolderPath, "train-labels-idx1-ubyte.gz");
        string uncompressedTrainLabels = Path.Combine(uncompressedFolderPath, "train-labels.idx1-ubyte");

        if (File.Exists(uncompressedTrainImages) == false || File.Exists(uncompressedTrainLabels) == false)
        {
            Decompress(compressedTrainImages, uncompressedTrainImages);
            Decompress(compressedTrainLabels, uncompressedTrainLabels);
        }

        string compressedTestImages = Path.Combine(compressedFolderPath, "t10k-images-idx3-ubyte.gz");
        string uncompressedTestImages = Path.Combine(uncompressedFolderPath, "t10k-images.idx3-ubyte");

        string compressedTestLabels = Path.Combine(compressedFolderPath, "t10k-labels-idx1-ubyte.gz");
        string uncompressedTestLabels = Path.Combine(uncompressedFolderPath, "t10k-labels.idx1-ubyte");

        if (File.Exists(uncompressedTestImages) == false || File.Exists(uncompressedTestImages) == false)
        {
            Decompress(compressedTestImages, uncompressedTestImages);
            Decompress(compressedTestLabels, uncompressedTestLabels);
        }

        ImageData[] trainingData = LoadImagesAndLabels(uncompressedTrainImages, uncompressedTrainLabels);
        ImageData[] testData = LoadImagesAndLabels(uncompressedTestImages, uncompressedTestLabels);

        cachedTestData = testData;
        cachedTrainingData = trainingData;

        return (trainingData, testData);
    }

    private static void Decompress(string compressedFilePath, string uncompressedFilePath)
    {
        FileInfo fileToDecompress = new FileInfo(compressedFilePath);
        using FileStream originalFileStream = fileToDecompress.OpenRead();

        using FileStream decompressedFileStream = File.Create(uncompressedFilePath);
        using GZipStream decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress);
        decompressionStream.CopyTo(decompressedFileStream);
    }

    /// <remarks>
    /// https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/
    /// </remarks>
    private static ImageData[] LoadImagesAndLabels(string imagePaths, string labelPaths)
    {
        // File format images
        // [offset] [type]          [description]
        // 0000     32 bit integer  magic number
        // 0004     32 bit integer  number of images
        // 0008     32 bit integer  number of rows
        // 0012     32 bit integer  number of columns
        // 0016     unsigned byte   pixel
        // 0017     unsigned byte   pixel
        // ...
        // xxxx     unsigned byte   pixel

        FileStream fileStreamImages = new FileStream(imagePaths, FileMode.Open, FileAccess.Read);
        BinaryReader binaryReaderImages = new BinaryReader(fileStreamImages);
        int magicNumberImages = ReadInt32(binaryReaderImages);
        int numberOfImages = ReadInt32(binaryReaderImages);
        int numberOfRows = ReadInt32(binaryReaderImages);
        int numberOfColumns = ReadInt32(binaryReaderImages);

        // File format labels
        // [offset] [type]          [description]
        // 0000     32 bit integer  magic number (MSB first)
        // 0004     32 bit integer  number of labels
        // 0008     unsigned byte   label
        // 0009     unsigned byte   label
        // ...
        // xxxx     unsigned byte   label

        FileStream fileStreamLabels = new FileStream(labelPaths, FileMode.Open, FileAccess.Read);
        BinaryReader binaryReaderLabels = new BinaryReader(fileStreamLabels);
        int magicNumberLabels = ReadInt32(binaryReaderLabels);
        int numberOfLabels = ReadInt32(binaryReaderLabels);

        if (numberOfImages != numberOfLabels)
        { throw new InvalidOperationException("Number of labels and images does not match."); }

        ImageData[] imageDatas = new ImageData[numberOfImages];
        for (int i = 0; i < numberOfImages; i++)
        {
            byte label = binaryReaderLabels.ReadByte();
            byte[,] pixels = new byte[numberOfRows, numberOfColumns];
            for (int y = 0; y < numberOfColumns; y++)
            {
                for (int x = 0; x < numberOfRows; x++)
                {
                    pixels[x, y] = binaryReaderImages.ReadByte();
                }
            }

            imageDatas[i] = new ImageData(label, pixels);
        }
        return imageDatas;
    }

    /// <remarks>
    /// Needed because: 'All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header. '
    /// See: https://stackoverflow.com/questions/20967088/what-did-i-do-wrong-with-parsing-mnist-dataset-with-binaryreader-in-c
    /// </remarks>
    private static int ReadInt32(BinaryReader binaryReader)
    {
        var bytes = binaryReader.ReadBytes(sizeof(int));
        if (BitConverter.IsLittleEndian) { Array.Reverse(bytes); }
        return BitConverter.ToInt32(bytes, 0);
    }
}
