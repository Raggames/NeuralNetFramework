using System;
using System.Collections.Generic;
using NeuralNetwork.Scripts.NetToolbox;
using UnityEngine;

namespace NeuralNetwork.Scripts.Data
{
    [Serializable]
    public class NetData
    {
        // Data Settings
        //===========================================================================================
        public bool HasData;
        public bool NewTraining;
        public int autoSaveIncrement;
        public string NeuralNetworkName;
        // ADN
        //===========================================================================================
        public double StartTrainingRate;
        public double[] InstanceWeights;
        public int DNAVersion;
        public List<NetLossParameter> PerformanceSolvers = new List<NetLossParameter>();
        public double PerformanceCoefficient;
        public double NetworkTrainingRate;
        //Architecture
        //===========================================================================================
        public int InputLayerNeurons;
        public List<int[]> HiddensAndNeurons = new List<int[]>();
        public int OutputLayerNeurons;

    }
}