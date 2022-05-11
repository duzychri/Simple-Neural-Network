using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

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
            return SerializeBinary(neuralNetworkData);
        }

        /// <summary>
        /// Deserializes a byte array into a <see cref="NeuralNetwork"/>.
        /// </summary>
        /// <param name="value">The byte array to deserialize.</param>
        /// <returns>The deserialized <see cref="NeuralNetwork"/>.</returns>
        public static NeuralNetwork DeserializeFromBytes(byte[] value)
        {
            NeuralNetworkData neuralNetworkData = DeserializeBinary<NeuralNetworkData>(value);
            return ConvertToNeuralNet(neuralNetworkData);
        }

        #endregion Api Methods

        #region Utility Methods

        private static byte[] SerializeBinary<T>(T value)
        {
            DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(T));
            using (MemoryStream compressedStream = new MemoryStream())
            using (MemoryStream serializedStream = new MemoryStream())
            using (DeflateStream deflateStream = new DeflateStream(compressedStream, CompressionMode.Compress, true))
            {
                serializer.WriteObject(serializedStream, value);
                serializedStream.Position = 0;
                serializedStream.CopyTo(deflateStream);
                deflateStream.Close();
                return compressedStream.ToArray();
            }
        }

        private static T DeserializeBinary<T>(byte[] value)
        {
            DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(T));
            using (MemoryStream inputStream = new MemoryStream(value))
            using (MemoryStream decompressedStream = new MemoryStream())
            using (DeflateStream deflateStream = new DeflateStream(inputStream, CompressionMode.Decompress))
            {
                deflateStream.CopyTo(decompressedStream);
                decompressedStream.Position = 0;
                return (T)serializer.ReadObject(decompressedStream);
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