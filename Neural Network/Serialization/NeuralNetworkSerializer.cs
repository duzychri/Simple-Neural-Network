using System.IO;
using System.IO.Compression;
using System.Linq;

namespace Neural_Network
{
    /// <summary>
    /// Exposes methods for the serialization and deserialization of a neural network.
    /// </summary>
    public static class NeuralNetworkSerializer
    {
        #region Api Methods

        /// <summary>
        /// Serializes the <see cref="NeuralNetwork"/> into a byte array.
        /// </summary>
        /// <param name="neuralNet">The <see cref="NeuralNetwork"/> to serialize.</param>
        /// <returns>The <see cref="NeuralNetwork"/> as a byte array.</returns>
        public static byte[] SerializeAsBytes(NeuralNetwork neuralNet)
        {
            NeuralNetworkData neuralNetworkData = ConvertToData(neuralNet);
            return SerializeBytes(neuralNetworkData);
        }

        /// <summary>
        /// Deserializes a byte array into a <see cref="NeuralNetwork"/>.
        /// </summary>
        /// <param name="value">The byte array to deserialize.</param>
        /// <returns>The deserialized <see cref="NeuralNetwork"/>.</returns>
        public static NeuralNetwork DeserializeFromBytes(byte[] value)
        {
            NeuralNetworkData neuralNetworkData = DeserializeBytes<NeuralNetworkData>(value);
            return ConvertToNeuralNet(neuralNetworkData);
        }

        #endregion Api Methods

        #region Utility Methods

        private static byte[] SerializeBytes<T>(T value)
        {
            using (MemoryStream compressedStream = new MemoryStream())
            using (MemoryStream serializedStream = new MemoryStream())
            using (GZipStream zipStream = new GZipStream(compressedStream, CompressionMode.Compress, true))
            {
                ProtoBuf.Serializer.Serialize(serializedStream, value);
                serializedStream.Position = 0;
                serializedStream.CopyTo(zipStream);
                zipStream.Close();
                return compressedStream.ToArray();
            }
        }

        private static T DeserializeBytes<T>(byte[] value)
        {
            using (MemoryStream inputStream = new MemoryStream(value))
            using (MemoryStream decompressedStream = new MemoryStream())
            using (GZipStream zipStream = new GZipStream(inputStream, CompressionMode.Decompress))
            {
                zipStream.CopyTo(decompressedStream);
                decompressedStream.Position = 0;
                T result = ProtoBuf.Serializer.Deserialize<T>(decompressedStream);
                return result;
            }
        }

        private static NeuralNetwork ConvertToNeuralNet(NeuralNetworkData neuralNetworkData)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork
            {
                InputSize = neuralNetworkData.InputSize,
                OutputSize = neuralNetworkData.Layers.Length,
                Layers = neuralNetworkData.Layers.Select((layer, index) => new NeuralLayer(index, layer.ActivationFunction)
                {
                    Neurons = layer.Neurons.Select(neuron => new Neuron
                    {
                        LayerIndex = index,
                        Bias = neuron.Bias,
                        Weights = neuron.Weights
                    }).ToArray()
                }).ToArray()
            };
            return neuralNetwork;
        }

        private static NeuralNetworkData ConvertToData(NeuralNetwork neuralNet)
        {
            NeuralNetworkData neuralNetworkData = new NeuralNetworkData
            {
                InputSize = neuralNet.InputSize,
                Layers = neuralNet.Layers.Select(layer => new NeuralLayerData
                {
                    ActivationFunction = layer.ActivationFunction,
                    Neurons = layer.Select(neuron => new NeuronData
                    {
                        Bias = neuron.Bias,
                        Weights = neuron.Weights
                    }).ToArray()
                }).ToArray(),
            };
            return neuralNetworkData;
        }

        #endregion Utility Methods
    }
}