using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    
    public class NetLayer
    {
        [HideInInspector] public NeuralNet NeuralNet;
        public enum LayerType
        {
            Input,
            Hidden,
            Output,
        }
        [HideInInspector] public double LayerBias;
        public LayerType ThisLayerType;
        public List<NetNeuron> NeuronsInLayer = new List<NetNeuron>();
       
        public EActivationFunctionType ActivationFunction;
        public enum EActivationFunctionType
        {
            Identity,
            Sigmoid,
            Tanh,
            Softmax,
            Linear,
            Boolean,
            Average,
            AverageForcePositive,

        }
    }

}