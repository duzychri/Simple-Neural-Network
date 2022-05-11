using System;
using System.Runtime.Serialization;

namespace Neural_Network
{
    [Serializable, DataContract]
    internal class NeuralLayerData
    {
        [DataMember]
        public NeuronData[] Neurons;
        [DataMember]
        public ActivationFunction ActivationFunction;
    }
}