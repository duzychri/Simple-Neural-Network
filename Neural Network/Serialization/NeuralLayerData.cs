using System;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, ProtoContract]
    internal class NeuralLayerData
    {
        [ProtoMember(1)]
        public NeuronData[] Neurons;
        [ProtoMember(2)]
        public ActivationFunction ActivationFunction;
    }
}