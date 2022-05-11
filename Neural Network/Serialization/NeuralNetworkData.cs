using System;
using System.Runtime.Serialization;

namespace Neural_Network
{
    [Serializable, DataContract]
    internal class NeuralNetworkData
    {
        [DataMember]
        public int InputSize;
        [DataMember]
        public NeuralLayerData[] Layers;
    }
}