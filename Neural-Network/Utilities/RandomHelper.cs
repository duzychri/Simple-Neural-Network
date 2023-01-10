using System;
using System.Threading;

namespace Neural_Network
{
    internal static class RandomHelper
    {
        private static readonly ThreadLocal<Random> localRandom = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));

        public static double NextDouble() => localRandom.Value.NextDouble();
        public static int Next(int maxValue) => localRandom.Value.Next(maxValue);
        public static double GetSmallRandomNumber() => (.0009 * NextDouble() + .0001) * (Next(2) == 0 ? -1 : 1);
    }
}