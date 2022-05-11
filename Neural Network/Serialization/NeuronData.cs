using System;
using System.Runtime.Serialization;

namespace Neural_Network
{
    [Serializable, DataContract]
    internal class NeuronData
    {
        [DataMember]
        public double Bias;
        [DataMember]
        public double[] Weights;
    }
}