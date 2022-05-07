using System;
using System.Runtime.Serialization;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, DataContract, ProtoContract]
    internal class NeuralLayerData
    {
        [DataMember, ProtoMember(1)]
        public NeuronData[] Neurons;
        [DataMember, ProtoMember(2)]
        public ActivationFunction ActivationFunction;
    }
}