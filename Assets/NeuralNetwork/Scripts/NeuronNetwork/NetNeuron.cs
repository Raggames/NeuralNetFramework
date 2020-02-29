using System;
using System.Collections.Generic;
using NeuralNetwork.Utils;
using UnityEngine;

namespace NeuralNetwork
{
  
     public class NetNeuron
    {
        public string NeuronID = "Neuron";
        public int PreviousNeurons;
        public int ForwardNeurons;
        public int OutputIndex;
       
        [HideInInspector] public List<double> Inputs = new List<double>();
        [HideInInspector] public List<double> Weights = new List<double>();
        [HideInInspector] public List<NetNeuron> ConnectedNeuronsList = new List<NetNeuron>();
        [HideInInspector] public NetLayer NetLayer;
        private int inputsReceivedIndex;
        
        public void ReceiveInput(double receivedValue)
        {
            Inputs.Add(receivedValue);
            inputsReceivedIndex++;
            if (inputsReceivedIndex == PreviousNeurons)
            {
                ComputeAndTransmit(receivedValue);
                inputsReceivedIndex = 0;
            }
        }
        
        public void ComputeAndTransmit(double inputsIfNeuronIsInputLayer = 0)
        {
            if (NetLayer.ThisLayerType == NetLayer.LayerType.Input)
            {
               double computedValue = CalculateWeight(Inputs, PreviousNeurons, NetLayer.LayerBias);
                    foreach (var neuron in ConnectedNeuronsList)
                    {
                       // neuron.Inputs.Add(computedValue);
                        //neuron.ComputeAndTransmit();
                        neuron.ReceiveInput(computedValue);
                    }
            }
            if (NetLayer.ThisLayerType == NetLayer.LayerType.Hidden)
            {
                    double computedValue = CalculateWeight(Inputs, PreviousNeurons, NetLayer.LayerBias);
                    foreach (var neuron in ConnectedNeuronsList)
                    {
                       neuron.ReceiveInput(computedValue);
                    }
            }
            if (NetLayer.ThisLayerType == NetLayer.LayerType.Output)
            {
                    double computedValue = CalculateOutput(Inputs, PreviousNeurons, NetLayer.LayerBias);
                    NetLayer.NeuralNet.ComputeNeuralNetOutput(computedValue, OutputIndex);
            }
            Inputs.Clear();
        }

        bool CheckInputs(List<double> input_values, int previousNeurons)
        {
            return input_values.Count >= previousNeurons;
        }

        double CalculateOutput(List<double> input_values, int previousNeurons, double bias)
        {
            double computedOutput = 0;
            for (int i = 0; i < previousNeurons; i++)
            {
                computedOutput += input_values[i] * Weights[i];
            }
            computedOutput += bias;
            
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Sigmoid)
            {
                
                computedOutput = NeuralMathCompute.Sigmoid(computedOutput);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Softmax)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Linear)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                computedOutput = NeuralMathCompute.Linear(computedOutput);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Boolean)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                computedOutput = NeuralMathCompute.Boolean(computedOutput);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Average)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                computedOutput /= (previousNeurons);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.AverageForcePositive)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                computedOutput /= previousNeurons;
                if (computedOutput < 0) computedOutput = -computedOutput;
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Tanh)
            {
                for (int i = 0; i < previousNeurons; i++)
                {
                    computedOutput += input_values[i] * Weights[i];
                }
                computedOutput += bias;
                computedOutput = Math.Tanh(computedOutput);
            }
            return computedOutput;
        }
        double CalculateWeight(List<double> input_values, int previousNeurons, double bias)
        {
            double computedValue = 0;
            for (int i = 0; i < previousNeurons; i++)
            {
                computedValue += input_values[i] * Weights[i];
            }
            computedValue += bias;
            if (NetLayer.ActivationFunction == NetLayer.EActivationFunctionType.Identity)
            {
                
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Sigmoid)
            {
                computedValue = NeuralMathCompute.Sigmoid(computedValue);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Linear)
            {
                computedValue = NeuralMathCompute.Linear(computedValue);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Boolean)
            {
                computedValue = NeuralMathCompute.Boolean(computedValue);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Average)
            {
                computedValue /= (previousNeurons);
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.AverageForcePositive)
            {
                computedValue /= previousNeurons;
                if (computedValue < 0) computedValue = -computedValue;
            }
            if (NetLayer.ActivationFunction == NeuralNetwork.NetLayer.EActivationFunctionType.Tanh)
            {
                computedValue = Math.Tanh(computedValue);
            }
            return computedValue;
        }
        
    }
}