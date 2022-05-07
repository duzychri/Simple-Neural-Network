using System;
using System.Runtime.Serialization;
using ProtoBuf;

namespace Neural_Network
{
    [Serializable, DataContract, ProtoContract]
    internal class NeuralNetworkData
    {
        [DataMember, ProtoMember(1)]
        public int InputSize;
        [DataMember, ProtoMember(2)]
        public NeuralLayerData[] Layers;
    }
}