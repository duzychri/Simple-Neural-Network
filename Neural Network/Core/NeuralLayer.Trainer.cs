using System.Collections.Generic;
using static Neural_Network.ActivationFunctionImplementations;

namespace Neural_Network
{
    internal partial class NeuralLayer : IEnumerable<Neuron>, IReadOnlyList<Neuron>
    {
        internal class Trainer
        {
            public readonly int index;
            public readonly NeuralLayer layer;

            public readonly double[] outputs;
            public readonly double[] totalInputs;

            public readonly double[] inputVotes;
            public readonly double[] outputVotes;

            public SlopeDelegate SlopeFunction => layer.SlopeDelegate;
            public ActivationDelegate ActivationFunction => layer.ActivationDelegate;

            public Trainer(int index, NeuralLayer layer)
            {
                this.index = index;
                this.layer = layer;

                outputs = new double[layer.Count];
                totalInputs = new double[layer.Count];

                inputVotes = new double[layer.Count];
                outputVotes = new double[layer.Count];
            }

            public double[] Fire(double[] inputValues)
            {
                for (int n = 0; n < layer.Neurons.Length; n++)
                {
                    Neuron neuron = layer.Neurons[n];
                    neuron.Fire(inputValues, layer.ActivationDelegate, out totalInputs[n], out outputs[n]);
                }
                return outputs;
            }

            #region IEnumerable & IReadOnlyList

            public int Count => layer.Count;
            public Neuron this[int index] => layer[index];

            public IEnumerator<Neuron> GetEnumerator() => ((IEnumerable<Neuron>)layer).GetEnumerator();

            #endregion IEnumerable & IReadOnlyList
        }
    }
}