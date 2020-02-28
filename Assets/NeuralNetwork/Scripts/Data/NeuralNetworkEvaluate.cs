using System;
using System.Collections.Generic;
using NeuralNetwork.Scripts.Data;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetworkEvaluate
    {
        public List<double> InstanceWeights;
        public List<double> InstanceBiases;

        public List<NeuralNetworkPerformanceSolver> PerformanceSolvers;
        public double PerformanceCoefficient;
        
    }
}