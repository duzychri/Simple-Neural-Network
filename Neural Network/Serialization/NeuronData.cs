using System;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, ProtoContract]
    internal class NeuronData
    {
        [ProtoMember(1)]
        public double Bias;
        [ProtoMember(2)]
        public double[] Weights;
    }
}