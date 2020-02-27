using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork.Scripts.Data
{
    [Serializable]
    public class NetData
    {
        public bool HasData;
        public string NeuralNetworkName;
        public double StartTrainingRate;
        public NeuralNet.DNA NeuralNetworkDna = new NeuralNet.DNA();
        public double NetworkTrainingRate;
        public List<double> NetworkBestResults = new List<double>();
        public double NotationCoefficient;
        public int DNAVersion;


    }
}