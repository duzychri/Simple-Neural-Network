using System;
using System.Linq;
using System.Diagnostics;

namespace Neural_Network
{
    /// <summary>
    /// A neural network that allows multiple perceptron layers.
    /// </summary>
    [DebuggerDisplay("Layer Count = {Count}"), DebuggerTypeProxy(typeof(NeuralNetDebugView))]
    public partial class NeuralNetwork
    {
        /// <summary>
        /// The amount of input values into the network.
        /// </summary>
        public int InputSize { get; internal set; }
        /// <summary>
        /// The amount of outputs from the network.
        /// </summary>
        public int OutputSize { get; internal set; }

        internal NeuralLayer[] Layers { get; set; }

        internal NeuralNetwork() { }

        /// <summary>
        /// Creates a new a neural network. 
        /// </summary>
        /// <param name="inputSize">The amount of input values into the network.</param>
        /// <param name="layers">The number of neurons and the activation functions for each layer.</param>
        /// <exception cref="ArgumentException">The size of input is 0 or lower.</exception>
        /// <exception cref="ArgumentException">The layers array is empty.</exception>
        /// <exception cref="ArgumentNullException">The layers array is null.</exception>
        public NeuralNetwork(int inputSize, LayerData[] layers)
        {
            if (inputSize <= 0) { throw new ArgumentException($"The {nameof(inputSize)} has to larger than 0.", nameof(inputSize)); }
            if (layers == null) { throw new ArgumentNullException(nameof(layers)); }
            if (layers.Length <= 0) { throw new ArgumentException("The amount of layers has to larger than 0.", nameof(layers)); }

            InputSize = inputSize;
            OutputSize = layers.Last().NeuronCount;

            this.Layers = new NeuralLayer[layers.Length];
            for (int l = 0; l < layers.Length; l++)
            {
                LayerData data = layers[l];
                int inputsCount = l == 0 ? inputSize : layers[l - 1].NeuronCount;

                Neuron[] neurons = Enumerable.Range(0, data.NeuronCount)
                    .Select(n => new Neuron(inputsCount, l))
                    .ToArray();

                this.Layers[l] = new NeuralLayer(l, neurons, data.ActivationFunction);
            }
        }

        /// <summary>
        /// Fires the network with the specified input values and return the output of the last layer.
        /// </summary>
        /// <param name="inputValues">The input values into the input layer.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException">The input value array is null.</exception>
        /// <exception cref="ArgumentException">The input value array doesn't have the correct input value count.</exception>
        public double[] Fire(double[] inputValues)
        {
            if (inputValues == null)
            { throw new ArgumentNullException(nameof(inputValues)); }
            if (InputSize != inputValues.Length)
            { throw new ArgumentException("The amount of input values needs to be equal to the amount of neurons in the first layer.", nameof(inputValues)); }

            double[] outputs = null;
            for (int n = 0; n < Layers.Length; n++)
            {
                NeuralLayer layer = Layers[n];
                outputs = layer.Fire(inputValues);
                inputValues = outputs;
            }
            return outputs;
        }

        /// <summary>
        /// Creates a new trainer for a <see cref="NeuralNetwork"/>.
        /// </summary>
        /// <param name="epochs">The amount of epochs to train.</param>
        /// <param name="learningRate">The strengh at which a perceptron is adjusted while learning.</param>
        /// <returns>A new trainer for a <see cref="NeuralNetwork"/>.</returns>
        /// <exception cref="ArgumentException">The amount of epochs or learning rate is smaller or equal to one.</exception>
        public NeuralNetworkTrainer GetTrainer(int epochs = 10, double learningRate = 0.0001)
        {
            if (epochs <= 0)
            { throw new ArgumentException("The amount of epochs needs to be larger than 0.", nameof(epochs)); }
            if (learningRate <= 0)
            { throw new ArgumentException("The learningRate needs to be larger than 0.", nameof(epochs)); }

            return new NeuralNetworkTrainer(this, epochs, learningRate);
        }

        #region Debug View

        internal sealed class NeuralNetDebugView
        {
            private readonly NeuralNetwork neuralNetwork;

            public NeuralNetDebugView(NeuralNetwork neuralNetwork)
            {
                if (neuralNetwork is null)
                { throw new ArgumentNullException(nameof(neuralNetwork)); }

                this.neuralNetwork = neuralNetwork;
            }

            [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
            public NeuralLayer[] Layers { get { return neuralNetwork.Layers; } }
        }

        #endregion Debug View
    }
}