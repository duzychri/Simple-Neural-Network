using System;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, ProtoContract]
    internal class NeuralNetworkData
    {
        [ProtoMember(1)]
        public int InputSize;
        [ProtoMember(2)]
        public NeuralLayerData[] Layers;
    }
}