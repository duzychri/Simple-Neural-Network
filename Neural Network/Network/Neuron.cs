using System;
using System.Text;
using System.Linq;
using System.Diagnostics;
using static Neural_Network.ActivationFunctionImplementations;

namespace Neural_Network
{
    [DebuggerDisplay("{ToString()}")]
    internal class Neuron
    {
        internal double Bias;
        internal double[] Weights;

        internal int LayerIndex { get; set; }

        internal readonly object BiasLock = new object();

        internal Neuron()
        { }

        internal Neuron(int inputsCount, int layerIndex)
        {
            this.LayerIndex = layerIndex;
            if (inputsCount <= 0) { throw new ArgumentException("The amount of inputs needs to be larger than 0.", nameof(inputsCount)); }
            Bias = RandomHelper.GetSmallRandomNumber();
            Weights = Enumerable.Range(0, inputsCount).Select(n => RandomHelper.GetSmallRandomNumber()).ToArray();
        }

        internal double Fire(double[] inputs, ActivationDelegate activationFunction)
        {
            return Fire(inputs, activationFunction, out double _);
        }

        internal double Fire(double[] inputs, ActivationDelegate activationFunction, out double totalInput)
        {
            Fire(inputs, activationFunction, out totalInput, out double output);
            return output;
        }

        internal void Fire(double[] inputs, ActivationDelegate activationFunction, out double totalInput, out double output)
        {
            totalInput = Bias;
            for (int n = 0; n < inputs.Length; n++)
            { totalInput += inputs[n] * Weights[n]; }

            output = activationFunction(totalInput);
        }

        //public void Learn(double[] inputs, double expectedOutput, double learningRate)
        //{
        //    double output = Fire(inputs, out double totalInput);

        //    double outputVotes = expectedOutput - output;
        //    double slope = slopeFunction(totalInput);
        //    double inputVotes = outputVotes * slope;

        //    var adjustment = inputVotes * learningRate;

        //    bias += adjustment;
        //    for (int n = 0; n < weights.Length; n++)
        //    { weights[n] += adjustment * inputs[n]; }
        //}

        #region Overrides

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append($"Bias = {Bias:0.000}, Weights = {{ ");
            for (int n = 0; n < Weights.Length; n++)
            {
                if (n == 0)
                { sb.Append($"{Weights[n]:0.000}"); }
                else
                { sb.Append($", {Weights[n]:0.000}"); }

                if (n > 5)
                {
                    sb.Append(", ...");
                    break;
                }
            }
            sb.Append(" }");
            return sb.ToString();
        }

        #endregion Overrides
    }
}