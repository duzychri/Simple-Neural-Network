#define PARALLEL

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Neural_Network
{
    /// <summary>
    /// A trainer for a <see cref="NeuralNetwork"/>.
    /// </summary>
    public class NeuralNetworkTrainer
    {
        /// <summary>
        /// The network to train.
        /// </summary>
        public NeuralNetwork Network { get; }

        /// <summary>
        /// The amount of epochs to train.
        /// </summary>
        public int Epochs { get; }
        /// <summary>
        /// The strengh at which a perceptron is adjusted while learning.
        /// </summary>
        public double LearningRate { get; }

        /// <summary>
        /// Creates a new trainer for a <see cref="NeuralNetwork"/>.
        /// </summary>
        /// <param name="neuralNet">The network to train.</param>
        /// <param name="epochs">The amount of epochs to train.</param>
        /// <param name="learningRate">The strengh at which a perceptron is adjusted while learning.</param>
        public NeuralNetworkTrainer(NeuralNetwork neuralNet, int epochs = 10, double learningRate = 0.001)
        {
            Network = neuralNet;
            Epochs = epochs;
            LearningRate = learningRate;
        }

        /// <summary>
        /// Tests the neural networks accuracy.
        /// </summary>
        /// <param name="testingSamples">The samples to use while testing.</param>
        /// <param name="isOutputCorrectFunction">The function that determines if the output of the network is correct.</param>
        /// <returns>The accuracy of the neural network.</returns>
        public double Test(TrainingSample[] testingSamples, Func<double[], double[], bool> isOutputCorrectFunction)
        {
            int bad = 0, good = 0;
            foreach (var sample in testingSamples)
            {
                double[] outputValues = Network.Fire(sample.Input);
                if (isOutputCorrectFunction(outputValues, sample.ExpectedOutput))
                { good++; }
                else
                { bad++; }
            }
            return (double)good / (good + bad);
        }

        /// <summary>
        /// Trains the neural network with a number of samples.
        /// </summary>
        /// <param name="trainingSamples">The samples to train the network with.</param>
        /// <exception cref="ArgumentNullException">The training samples input array is null.</exception>
        /// <exception cref="ArgumentException">The training samples input or output size doesn't fit the neural networks input and output size.</exception>
        public void Train(TrainingSample[] trainingSamples)
        {
            foreach (TrainingSample sample in trainingSamples)
            {
                if (sample.Input == null)
                { throw new ArgumentNullException(nameof(trainingSamples), "The inputs of the sample need to exist."); }
                if (Network.InputSize != sample.Input.Length)
                { throw new ArgumentException("The amount of input values needs to be equal to the input size of the network.", nameof(trainingSamples)); }
                if (Network.OutputSize != sample.ExpectedOutput.Length)
                { throw new ArgumentException("The amount of exptected output values needs to be equal to the output size of the network.", nameof(trainingSamples)); }
            }

            double currLearningRate = LearningRate;
            for (int n = 0; n < Epochs; n++)
            {
                currLearningRate *= .9;
                trainingSamples.Shuffle();

                ThreadLocal<NetworkTrainer> trainer = new ThreadLocal<NetworkTrainer>(() => new NetworkTrainer(Network));

#if PARALLEL
                Parallel.ForEach(trainingSamples, sample =>
                {
                    trainer.Value.Train(sample, currLearningRate);
                });
#else
            foreach (TrainingSample sample in trainingSamples)
            {
                trainer.Value.Train(sample, currLearningRate);
            }
#endif
            }
        }
    }
}