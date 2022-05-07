using System;
using System.Runtime.Serialization;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, DataContract, ProtoContract]
    internal class NeuronData
    {
        [DataMember, ProtoMember(1)]
        public double Bias;
        [DataMember, ProtoMember(2)]
        public double[] Weights;
    }
}