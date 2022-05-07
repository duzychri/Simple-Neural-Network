using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using static Neural_Network.ActivationFunctionImplementations;

namespace Neural_Network
{
    [DebuggerDisplay("Neuron Count = {Count}"), DebuggerTypeProxy(typeof(NeuralLayerDebugView))]
    internal partial class NeuralLayer : IEnumerable<Neuron>, IReadOnlyList<Neuron>
    {
        internal Neuron[] Neurons { get; set; }
        internal ActivationFunction ActivationFunction { get; }

        private readonly int index;
        private readonly SlopeDelegate SlopeDelegate;
        private readonly ActivationDelegate ActivationDelegate;

        internal NeuralLayer(int index, ActivationFunction activationFunction)
        {
            this.index = index;
            (ActivationDelegate, SlopeDelegate) = activationFunction.GetFunctions();
        }

        internal NeuralLayer(int index, Neuron[] neurons, ActivationFunction activationFunction)
        {
            this.index = index;
            this.ActivationFunction = activationFunction;
            this.Neurons = neurons ?? throw new ArgumentNullException(nameof(neurons));
            (ActivationDelegate, SlopeDelegate) = activationFunction.GetFunctions();
        }

        internal double[] Fire(double[] inputValues, double[] outputValues = null)
        {
            if (outputValues == null)
            { outputValues = new double[Neurons.Length]; }

            for (int n = 0; n < Neurons.Length; n++)
            {
                Neuron neuron = Neurons[n];
                outputValues[n] = neuron.Fire(inputValues, ActivationDelegate);
            }

            return outputValues;
        }

        #region IEnumerable & IReadOnlyList

        public int Count => Neurons.Length;
        public Neuron this[int index] => Neurons[index];

        IEnumerator IEnumerable.GetEnumerator() => Neurons.GetEnumerator();
        public IEnumerator<Neuron> GetEnumerator() => ((IEnumerable<Neuron>)Neurons).GetEnumerator();

        #endregion IEnumerable & IReadOnlyList

        #region Debug View

        internal sealed class NeuralLayerDebugView
        {
            [DebuggerDisplay("{Value}", Name = "{Name,nq}")]
            public struct Property
            {
                public string Name { get; }
                public object Value { get; }

                public Property(string name, object value)
                {
                    Name = name;
                    Value = value;
                }
            }

            public NeuralLayerDebugView(NeuralLayer neuralLayer)
            {
                if (neuralLayer is null)
                { throw new ArgumentNullException(nameof(neuralLayer)); }

                Properties = new Property[neuralLayer.Neurons.Length + 2];
                Properties[0] = new Property("Index", neuralLayer.index);
                Properties[1] = new Property("Activation Function", neuralLayer.ActivationFunction);
                for (int n = 0; n < neuralLayer.Neurons.Length; n++)
                { Properties[n + 2] = new Property($"[{n}]", neuralLayer.Neurons[n]); }
            }

            [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
            public Property[] Properties { get; private set; }
        }

        #endregion Debug View
    }
}