using System;

namespace Neural_Network
{
    internal static class ActivationFunctionImplementations
    {
        public delegate double ActivationDelegate(double totalInput);
        public delegate double SlopeDelegate(double totalInput, double output);

        public static double ReluFunction(double totalInput)
        { return (totalInput >= 0d) ? totalInput : totalInput / 100d; }

        public static double ReluSlope(double totalInput, double output)
        { return (totalInput >= 0d) ? 1d : 0.01d; }

        public static double LogisticSigmoidFunction(double totalInput)
        { return 1d / (1d + Math.Exp(-totalInput)); }

        public static double LogisticSigmoidSlope(double totalInput, double output)
        { return output * (1d - output); ; }

        public static double HyperTanFunction(double totalInput)
        { return Math.Tanh(totalInput); }

        public static double HyperTanSlope(double totalInput, double output)
        { return 1d - output * output; }

        public static (ActivationDelegate, SlopeDelegate) GetFunctions(this ActivationFunction activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunction.ReLU:
                    return (ReluFunction, ReluSlope);
                case ActivationFunction.Sigmoid:
                    return (LogisticSigmoidFunction, LogisticSigmoidSlope);
                case ActivationFunction.HyperbolicTangent:
                    return (HyperTanFunction, HyperTanSlope);
                default:
                    throw new InvalidOperationException();
            }
        }
    }
}