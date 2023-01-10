using System.Text;
using System.Diagnostics;

namespace Neural_Network
{
    /// <summary>
    /// A sample to train a neural network on.
    /// </summary>
    [DebuggerDisplay("{ToString()}")]
    public class TrainingSample
    {
        /// <summary>
        /// The input into the sample.
        /// </summary>
        public double[] Input;
        /// <summary>
        /// The expected, correct output that the neural network should return.
        /// </summary>
        public double[] ExpectedOutput;

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("Input: ");
            if (Input.Length > 10)
            { sb.Append("..."); }
            else
            { sb.Append(string.Join(", ", Input)); }

            sb.Append("; Expected Output: ");
            if (ExpectedOutput.Length > 10)
            { sb.Append("..."); }
            else
            { sb.Append(string.Join(", ", ExpectedOutput)); }

            return sb.ToString();
        }
    }
}