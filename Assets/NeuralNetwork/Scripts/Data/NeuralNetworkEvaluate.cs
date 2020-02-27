using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetworkEvaluate
    {
        public List<double> InstanceWeights;
        public List<double> InstanceBiases;
        public double NotationCoefficient;
        public List<double> Results;
    }
}