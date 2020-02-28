using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork.Scripts.Data
{
    [Serializable]
    public class NetData
    {
        public bool HasData;
        public bool NewTraining;
        public string NeuralNetworkName;
        public double StartTrainingRate;
        public NeuralNet.DNA NeuralNetworkDna = new NeuralNet.DNA();
        public double NetworkTrainingRate;
        public List<NeuralNetworkPerformanceSolver> PerformanceSolvers = new List<NeuralNetworkPerformanceSolver>();
        public double PerformanceCoefficient;
        public int DNAVersion;


    }
}