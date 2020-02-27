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
                if (CheckInputs(Inputs, 1))
                {
                    double computedValue = CalculateWeight(Inputs, PreviousNeurons, NetLayer.LayerBias, NetLayer.NeuralNet.NeuralNetworkManager.ComputeActivationFunctionType);
                    foreach (var neuron in ConnectedNeuronsList)
                    {
                       // neuron.Inputs.Add(computedValue);
                        //neuron.ComputeAndTransmit();
                        neuron.ReceiveInput(computedValue);
                    }
                   
                }
            }
            if (NetLayer.ThisLayerType == NetLayer.LayerType.Hidden)
            {
                if (CheckInputs(Inputs, PreviousNeurons))
                {
                    double computedValue = CalculateWeight(Inputs, PreviousNeurons, NetLayer.LayerBias, NetLayer.NeuralNet.NeuralNetworkManager.ComputeActivationFunctionType);
                    foreach (var neuron in ConnectedNeuronsList)
                    {
                        //neuron.Inputs.Add(computedValue);
                       // neuron.ComputeAndTransmit();
                       neuron.ReceiveInput(computedValue);

                    }
                }
            }
            if (NetLayer.ThisLayerType == NetLayer.LayerType.Output)
            {
                if (CheckInputs(Inputs, PreviousNeurons))
                {
                    double computedValue = CalculateOutput(Inputs, PreviousNeurons, NetLayer.LayerBias, NetLayer.NeuralNet.NeuralNetworkManager.OutputActivationFunctionType);
                    NetLayer.NeuralNet.ComputeNeuralNetOutput(computedValue, OutputIndex);
                }
            }
            Inputs.Clear();
        }

        bool CheckInputs(List<double> input_values, int previousNeurons)
        {
            return input_values.Count >= previousNeurons;
        }

        double CalculateOutput(List<double> input_values, int previousNeurons, double bias, NeuralNetworkManager.EOutputActivationFunctionType activationFunctionType)
        {
            double computedOutput = 0;
            for (int i = 0; i < previousNeurons; i++)
            {
                computedOutput += input_values[i] * Weights[i];
            }
            computedOutput += bias;
            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.Sigmoid)
            {
                computedOutput = NeuralMathCompute.Sigmoid(computedOutput);
            }
            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.Softmax)
            {
                computedOutput = NeuralMathCompute.Softmax(computedOutput);
            }
            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.Linear)
            {
                computedOutput = NeuralMathCompute.Linear(computedOutput);
            }
            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.Boolean)
            {
                computedOutput = NeuralMathCompute.Boolean(computedOutput);
            }
            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.Average)
            {
                computedOutput /= (previousNeurons);
            }

            if (activationFunctionType == NeuralNetworkManager.EOutputActivationFunctionType.AverageForcePositive)
            {
                computedOutput /= previousNeurons;
                if (computedOutput < 0) computedOutput = -computedOutput;
            }
            return computedOutput;
        }
        double CalculateWeight(List<double> input_values, int previousNeurons, double bias, NeuralNetworkManager.EComputeActivationFunctionType activationFunctionType)
        {
            double computedValue = 0;
            for (int i = 0; i < previousNeurons; i++)
            {
                computedValue += input_values[i] * Weights[i];
            }
            computedValue += bias;
            //
            if (activationFunctionType == NeuralNetworkManager.EComputeActivationFunctionType.Sigmoid)
            {
                computedValue = NeuralMathCompute.Sigmoid(computedValue);
            }
            if (activationFunctionType == NeuralNetworkManager.EComputeActivationFunctionType.Softmax)
            {
                computedValue = NeuralMathCompute.Softmax(computedValue);
            }
            if (activationFunctionType == NeuralNetworkManager.EComputeActivationFunctionType.Linear)
            {
                computedValue = NeuralMathCompute.Linear(computedValue);
            }
            if (activationFunctionType == NeuralNetworkManager.EComputeActivationFunctionType.Boolean)
            {
                computedValue = NeuralMathCompute.Boolean(computedValue);
            }

            if (activationFunctionType == NeuralNetworkManager.EComputeActivationFunctionType.Average)
            {
                computedValue /= (previousNeurons);
            }
            return computedValue;
        }
        
    }
}