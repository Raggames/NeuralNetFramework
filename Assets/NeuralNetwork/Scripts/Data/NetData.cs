using System;
using System.Collections.Generic;
using NeuralNetwork.Scripts.NetToolbox;
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

        public double[] InstanceWeights;

        public List<NetLossParameter> PerformanceSolvers = new List<NetLossParameter>();
        public double PerformanceCoefficient;
        public double NetworkTrainingRate;

        public int DNAVersion;
        public int autoSaveIncrement;

        public int InputLayerNeurons;
        public List<int[]> HiddensAndNeurons = new List<int[]>();
        public int OutputLayerNeurons;

    }
}