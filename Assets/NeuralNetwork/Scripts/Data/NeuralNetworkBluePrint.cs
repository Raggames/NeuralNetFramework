using System;
using System.Collections.Generic;
using NeuralNetwork.Scripts.Data;
using UnityEngine;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetworkBluePrint 
    {
        public NetData NetData;
        public  List<NeuralNet.Layer> InputLayerConstruct;
        public List<NeuralNet.Layer> HiddenLayerConstruct = new List<NeuralNet.Layer>();
        public  List<NeuralNet.Layer> OutputLayerConstruct;
       
    }
}