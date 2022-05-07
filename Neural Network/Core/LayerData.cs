namespace Neural_Network
{
    /// <summary>
    /// The initialization information of a layer for a new neural network.
    /// </summary>
    public class LayerData
    {
        /// <summary>
        /// The amount of neurons in the layer.
        /// </summary>
        public int NeuronCount;
        /// <summary>
        /// The activator function used for the neurons of the layer.
        /// </summary>
        public ActivationFunction ActivationFunction;
    }
}
