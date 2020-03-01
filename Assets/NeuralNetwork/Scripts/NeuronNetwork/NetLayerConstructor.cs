using System;
using System.Collections.Generic;
using UnityEngine;
using Activator = NeuralNetwork.Scripts.NeuronActivatorFunctions.Activator;

namespace NeuralNetwork
{
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
        public Activator ActivatorFunction;
    }

}