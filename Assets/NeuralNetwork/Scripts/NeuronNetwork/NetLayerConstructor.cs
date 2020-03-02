using System;
using System.Collections.Generic;
using NeuralNetwork.Scripts.NeuronActivatorFunctions;
using UnityEngine;
using Activator = NeuralNetwork.Scripts.NeuronActivatorFunctions.Activator;

namespace NeuralNetwork
{
    [Serializable]
    public class NetLayerConstructor
    {
        public enum LayerType
        {
            Input,
            Hidden,
            Output,
        }
        public LayerType ThisLayerType;
        public int Neurons;
        public ActivatorType ActivatorFunction;
    }

}