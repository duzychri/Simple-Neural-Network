using System;
using System.Linq;

namespace Neural_Network
{
    internal class NetworkTrainer
    {
        public NeuralNetwork NeuralNet { get; }

        private readonly LayerTrainer[] layers;

        public NetworkTrainer(NeuralNetwork neuralNet)
        {
            NeuralNet = neuralNet ?? throw new ArgumentNullException(nameof(neuralNet));
            layers = NeuralNet.Layers.Select((l, i) => new LayerTrainer(i, l)).ToArray();
        }

        public void Train(TrainingSample trainingSample, double learingRate)
        {
            FeedForward(trainingSample.Input);
            CalculateVotes(trainingSample.ExpectedOutput);
            UpdateWeightsAndBiases(trainingSample.Input, learingRate);
        }

        private void FeedForward(double[] inputValues)
        {
            for (int n = 0; n < layers.Length; n++)
            {
                LayerTrainer layer = layers[n];
                inputValues = layer.Fire(inputValues);
            }
        }

        /// <summary>
        /// Calculates all the output and input votes.
        /// </summary>
        private void CalculateVotes(double[] desiredOutputValues)
        {
            // Step through layers in reverse.
            for (int r = layers.Length - 1; r >= 0; r--)
            {
                LayerTrainer layer = layers[r];
                bool isOutputLayer = r == layers.Length - 1;
                LayerTrainer previousLayer = isOutputLayer ? null : layers[r + 1];

                for (int n = 0; n < layer.Count; n++)
                {
                    Neuron neuron = layer[n];

                    // Calculate output votes.
                    if (isOutputLayer)
                    {
                        // For neurons in the output layer, the loss vs output slope = -error.
                        layer.outputVotes[n] = desiredOutputValues[n] - layer.outputs[n];
                    }
                    else
                    {
                        // For hidden neurons, the loss vs output slope = weighted sum of next layer's input slopes.
                        double outputVotes = 0;
                        for (int p = 0; p < previousLayer.Count; p++)
                        {
                            outputVotes += previousLayer.inputVotes[p] * previousLayer[p].Weights[n];
                        }

                        layer.outputVotes[n] = outputVotes;
                        //layer.outputVotes[index] = previousLayer.Sum(i => i.InputVotes * i.weights[index]);
                    }

                    // Calculate input votes.
                    // The loss vs input slope = loss vs output slope times activation function slope (chain rule).
                    layer.inputVotes[n] = layer.outputVotes[n] * layer.SlopeFunction(layer.totalInputs[n], layer.outputs[n]);
                }
            }
        }

        private void UpdateWeightsAndBiases(double[] inputValues, double learningRate)
        {
            // Updates weights and biases.
            for (int l = 0; l < layers.Length; l++)
            {
                LayerTrainer layer = layers[l];
                for (int n = 0; n < layer.Count; n++)
                {
                    Neuron neuron = layer[n];
                    // We can improve the training by scaling the learning rate by the layer index.
                    // double betterLearningRate = learningRate * (layers.Length - l);
                    UpdateWeightsAndBiasOfNeuron(neuron, layer.inputVotes[n], inputValues, learningRate);
                }

                if (l != layers.Length - 1)
                { inputValues = layer.outputs; }
            }
        }

        private static unsafe void UpdateWeightsAndBiasOfNeuron(Neuron neuron, double inputVotes, double[] inputValues, double learningRate)
        {
            double adjustment = inputVotes * learningRate;

#if DEBUG
            if (inputValues.Length != neuron.Weights.Length)
            { throw new InvalidOperationException(); }
#endif

            lock (neuron.BiasLock)
            { neuron.Bias += adjustment; }

            int max = neuron.Weights.Length;
            fixed (double* inputs = inputValues)
            fixed (double* weights = neuron.Weights)
            {
                lock (neuron.Weights)
                {
                    for (int i = 0; i < max; i++)
                    {
                        // Neuron.InputWeights [i] += adjustment * inputValues [i];
                        // Using pointers avoids bounds-checking and so reduces the time spent holding the lock.
                        *(weights + i) += adjustment * *(inputs + i);
                    }
                }
            }
        }
    }
}