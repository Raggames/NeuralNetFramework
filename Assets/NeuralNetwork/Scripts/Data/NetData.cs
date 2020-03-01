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
       
        public List<double> InstanceWeights = new List<double>();
        public List<double> InstanceBiases = new List<double>();

        public List<NeuralNetworkPerformanceSolver> PerformanceSolvers = new List<NeuralNetworkPerformanceSolver>();
        public double PerformanceCoefficient;
        public double NetworkTrainingRate;

        public int DNAVersion;
        public int autoSaveIncrement;

        public int InputLayerNeurons;
        public List<int[]> HiddensAndNeurons;
        public int OutputLayerNeurons;

    }
}