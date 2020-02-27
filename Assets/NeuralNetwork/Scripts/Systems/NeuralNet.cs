using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Utils;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NeuralNetwork
{
    [RequireComponent(typeof(NeuralNetworkComponent))]
    public class NeuralNet : MonoBehaviour
    {
        [Header("Neural Network Architecture")]
        
        public NeuralNetworkManager NeuralNetworkManager;
        public NeuralNetworkComponent NeuralNetworkComponent;

        [Header("Network Construction")] 
        public List<Layer> InputLayerConstruct = new List<Layer>(1);
        public List<Layer> HiddenLayerConstruct = new List<Layer>();
        public List<Layer> OutputLayerConstruct = new List<Layer>(1);
        
        public NetLayer InputLayer = new NetLayer();
        public List<NetLayer> HiddenLayers = new List<NetLayer>();
        public NetLayer OutputLayer = new NetLayer();
        
       
        [HideInInspector] public NetData _NetData;
        [Header("DNA")]
        public DNA InstanceDNA = new DNA();
        
        [Header("Network Execution")]
        public int InstanceID;
        public bool IsExecuting;
        public bool IsTraining;
        public int EpochsRemaining;
     
       
        public int NeuralNetConnections;
        [SerializeField] private List<double> _cycleResults = new List<double>();
        private int _countResultsEntry;
        private int _outputCount;
        
        private void CreateNetwork( List<Layer> input, List<Layer> hiddens,  List<Layer> output)
        {
            Debug.Log("Creating NeuralNet Network");
            NetLayer inputLayer = new NetLayer();
            inputLayer.ThisLayerType = NetLayer.LayerType.Input;
            inputLayer.LayerBias = input[0].LayerBias;
            InputLayer = inputLayer;
            for (int i = 0; i < input[0].NeuronCount; i++)
            {
                NetNeuron neuron = new NetNeuron();
                InputLayer.NeuronsInLayer.Add(neuron);
            }

            for (int i = 0; i < hiddens.Count; i++)
            {
                NetLayer netLayer = new NetLayer();
                netLayer.ThisLayerType = NetLayer.LayerType.Hidden;
                netLayer.LayerBias = hiddens[i].LayerBias;
                HiddenLayers.Add(netLayer);
                for (int j = 0; j < hiddens[i].NeuronCount; j++)
                {
                    NetNeuron neuron = new NetNeuron();
                    HiddenLayers[i].NeuronsInLayer.Add(new NetNeuron());
                }
            }
            NetLayer outputLayer = new NetLayer();
            outputLayer.ThisLayerType = NetLayer.LayerType.Output;
            outputLayer.LayerBias = output[0].LayerBias;
            OutputLayer = outputLayer;
            for (int i = 0; i < output[0].NeuronCount; i++)
            {
                NetNeuron neuron = new NetNeuron();
                OutputLayer.NeuronsInLayer.Add(new NetNeuron());
            }
            _outputCount = output[0].NeuronCount;
            for (int i = 0; i < _outputCount; i++)
            {
                _cycleResults.Add(0);
            }
        }
        public void InitializeNeuralNetwork(NeuralNetworkManager.ENetworkMode eNetworkMode, int epochs = 0, int instanceID = 0, NeuralNetworkManager neuralNetworkManager = null)
        {
            Debug.Log("Starting Initialisation");
            CreateNetwork(InputLayerConstruct, HiddenLayerConstruct, OutputLayerConstruct);
            NeuralNetworkManager = neuralNetworkManager;
            NeuralNetConnections = ComputeNumberOfWeights();
            InstanceID = instanceID;
            EpochsRemaining = epochs;
            InputLayer.NeuralNet = this;
            foreach (var hidden in HiddenLayers)
            {
                hidden.NeuralNet = this;
            }
            OutputLayer.NeuralNet = this;
            ///////////////////////////////////////////////            
            int IDIndex = 0;
            int LayerIndex = 0;
            // Input Layer to first Hidden
            foreach (var neuron in InputLayer.NeuronsInLayer)
            {
                neuron.NetLayer = InputLayer;
                neuron.ForwardNeurons = HiddenLayers[0].NeuronsInLayer.Count;
                neuron.NeuronID = "Input Neuron " + IDIndex.ToString();
                neuron.PreviousNeurons = 1; //Previous neurons is considered as external input (1 value) for the Input Layer Case
                for (int i = 0; i < HiddenLayers[0].NeuronsInLayer.Count; i++)
                {
                    neuron.ConnectedNeuronsList.Add(HiddenLayers[0].NeuronsInLayer[i]);
                }
                IDIndex++;
            }
            IDIndex = 0;
            //BetweenHiddens
            for (int i = 0; i < HiddenLayers.Count-1; i++)
            {
                for (int j = 0; j < HiddenLayers[i].NeuronsInLayer.Count; j++)
                {
                    HiddenLayers[i].NeuronsInLayer[j].ConnectedNeuronsList = HiddenLayers[i+1].NeuronsInLayer;
                }

                if (i == 0)
                {
                    foreach (var neuron in HiddenLayers[i].NeuronsInLayer)
                    {
                        neuron.NetLayer = HiddenLayers[i];
                        neuron.ForwardNeurons = HiddenLayers[i+1].NeuronsInLayer.Count;
                        neuron.PreviousNeurons = InputLayer.NeuronsInLayer.Count;
                        neuron.NeuronID = "Hidden " + LayerIndex + "Neuron " + IDIndex;
                        IDIndex++;
                    }
                }
                else
                {
                    foreach (var neuron in HiddenLayers[i].NeuronsInLayer)
                    {
                        neuron.NetLayer = HiddenLayers[i];
                        neuron.ForwardNeurons = HiddenLayers[i+1].NeuronsInLayer.Count;
                        neuron.PreviousNeurons = HiddenLayers[i - 1].NeuronsInLayer.Count;
                        neuron.NeuronID = "Hidden " + LayerIndex + "Neuron " + IDIndex;
                        IDIndex++;
                    }
                }
               
                IDIndex = 0;
                LayerIndex++;
            }
            //To Output
            IDIndex = 0;
            foreach (var neuron in HiddenLayers[HiddenLayers.Count-1].NeuronsInLayer)
            {
                neuron.ForwardNeurons = OutputLayer.NeuronsInLayer.Count;
                neuron.PreviousNeurons = HiddenLayers[HiddenLayers.Count - 2].NeuronsInLayer.Count;
                neuron.NetLayer = HiddenLayers[HiddenLayers.Count - 1];
                neuron.NeuronID = "Hidden " + LayerIndex + "Neuron " + IDIndex;
                IDIndex++;
            }
            IDIndex = 0;
            for (int j = 0; j < HiddenLayers[HiddenLayers.Count-1].NeuronsInLayer.Count; j++)
            {
                HiddenLayers[HiddenLayers.Count-1].NeuronsInLayer[j].ConnectedNeuronsList = OutputLayer.NeuronsInLayer;

                for (int i = 0; i < HiddenLayers[HiddenLayers.Count-1].NeuronsInLayer[j].ConnectedNeuronsList[i].ConnectedNeuronsList.Count; i++)
                {
                    HiddenLayers[HiddenLayers.Count - 1].NeuronsInLayer[j].ConnectedNeuronsList[i].ConnectedNeuronsList =
                        OutputLayer.NeuronsInLayer;
                }
                
            }
            foreach (var neuron in OutputLayer.NeuronsInLayer)
            {
                neuron.OutputIndex = IDIndex;
                neuron.NetLayer = OutputLayer;
                neuron.ForwardNeurons = 0;
                neuron.PreviousNeurons = HiddenLayers[HiddenLayers.Count-1].NeuronsInLayer.Count;
                neuron.NeuronID = neuron.NeuronID + " " + IDIndex;
                IDIndex++;
            }   
            //////////////////////////
            if (NeuralNetworkManager.NetworkMode == NeuralNetwork.NeuralNetworkManager.ENetworkMode.Train)
            {
                if(NeuralNetworkManager.NewTraining) SetDNARandom(NeuralNetConnections);
                if(!NeuralNetworkManager.NewTraining) SetDNAFromData(_NetData, NeuralNetworkManager.ForceGeneticsRandomization, NeuralNetworkManager.TrainingRate);
            }

            if (NeuralNetworkManager.NetworkMode == NeuralNetwork.NeuralNetworkManager.ENetworkMode.Train)
            {
                SetDNAFromData(_NetData, NeuralNetworkManager.EForceRandomization.No, NeuralNetworkManager.TrainingRate);
            }
            StartInstance(eNetworkMode);
           
        }
        private int ComputeNumberOfWeights()
        {
            int nbr = 0;
            nbr += (InputLayer.NeuronsInLayer.Count + 1) * HiddenLayers[0].NeuronsInLayer.Count;
            for (int i = 0; i < HiddenLayers.Count-1; i++)
            {
                if (i == HiddenLayers.Count - 2)
                {
                    nbr += (HiddenLayers[i].NeuronsInLayer.Count + 1) * OutputLayer.NeuronsInLayer.Count;
                } 
                else
                {
                    nbr += (HiddenLayers[i].NeuronsInLayer.Count + 1) * HiddenLayers[i + 1].NeuronsInLayer.Count;
                }
            }
            return nbr;
        }
        private double GetBias(NetLayer layer)
        {
            return layer.LayerBias;
        }
        private void SetDNARandom(int WeightsCount)  //On Network first Initialisation
        {
            DNA randomDna = new DNA();
            randomDna.Weights = new List<double>();
            randomDna.Biases = new List<double>();
            // In order : InputWeights, Input Bias, Hidden Weight/Hidden Bias (*hiddens), Output Weights, Output bias
            
            int index = 0;
            double weight = 0;
            for (int i = 0; i < WeightsCount*2; i++)
            {
                weight = Random.Range(0f, 1f);
                randomDna.Weights.Add(weight);
            }
            for (int i = 0; i < InputLayer.NeuronsInLayer.Count; i++)
            {
                InputLayer.NeuronsInLayer[i].Weights.Clear();
                InputLayer.NeuronsInLayer[i].Weights.Add(randomDna.Weights[index]);
                index++;
            }
            
            double inputBias = GetBias(InputLayer);
            randomDna.Biases.Add(inputBias);
            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                for (int j = 0; j < HiddenLayers[i].NeuronsInLayer.Count; j++)
                {
                    HiddenLayers[i].NeuronsInLayer[j].Weights.Clear();
                    for (int k = 0; k < HiddenLayers[i].NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        HiddenLayers[i].NeuronsInLayer[j].Weights.Add(randomDna.Weights[index]);
                        index++;
                    }
                    
                }
                double hiddenBias = GetBias(HiddenLayers[i]);
                randomDna.Biases.Add(hiddenBias);
            }
           
                for (int j = 0; j < OutputLayer.NeuronsInLayer.Count; j++)
                {
                    OutputLayer.NeuronsInLayer[j].Weights.Clear();
                    for (int k = 0; k < OutputLayer.NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        OutputLayer.NeuronsInLayer[j].Weights.Add(randomDna.Weights[index]);
                        index++;
                    }
                }
            double outputBias = GetBias(OutputLayer);
            randomDna.Biases.Add(outputBias);

            InstanceDNA = randomDna;
            Debug.Log(index);
        }
        private void SetDNAFromData(NetData netData, NeuralNetworkManager.EForceRandomization forceRandomization, double trainingRate)
        {
            if (netData.HasData)
            {
              
            if (forceRandomization == NeuralNetworkManager.EForceRandomization.No)
            {
                List<double> dataWeights = new List<double>();
                List<double> dataBiases = new List<double>();
                int weightIndex = 0;
                int biaseIndex = 0;
                dataWeights = netData.NeuralNetworkDna.Weights;
                dataBiases = netData.NeuralNetworkDna.Biases;
                InstanceDNA = netData.NeuralNetworkDna;
                
                for (int i = 0; i < InputLayer.NeuronsInLayer.Count; i++)
                {
                    InputLayer.NeuronsInLayer[i].Weights.Clear();
                    InputLayer.NeuronsInLayer[i].Weights.Add(dataWeights[weightIndex]);
                    weightIndex++;
                }
                InputLayer.LayerBias = dataBiases[biaseIndex];
                biaseIndex++;
                for (int i = 0; i < HiddenLayers.Count; i++)
                {
                    for (int j = 0; j < HiddenLayers[i].NeuronsInLayer.Count; j++)
                    {
                        HiddenLayers[i].NeuronsInLayer[j].Weights.Clear();
                        for (int k = 0; k < HiddenLayers[i].NeuronsInLayer[j].PreviousNeurons; k++)
                        {
                            HiddenLayers[i].NeuronsInLayer[j].Weights.Add(dataWeights[weightIndex]);
                            weightIndex++;
                        }
                    
                    }
                    HiddenLayers[i].LayerBias = dataBiases[biaseIndex];
                    biaseIndex++;
                }
                    for (int j = 0; j < OutputLayer.NeuronsInLayer.Count; j++)
                    {
                        OutputLayer.NeuronsInLayer[j].Weights.Clear();
                        for (int k = 0; k < OutputLayer.NeuronsInLayer[j].PreviousNeurons; k++)
                        {
                            OutputLayer.NeuronsInLayer[j].Weights.Add(dataWeights[weightIndex]);
                            weightIndex++;
                        }
                    }
                OutputLayer.LayerBias = dataBiases[biaseIndex];
            }

            if (forceRandomization == NeuralNetworkManager.EForceRandomization.Yes)
            {
                List<double> dataWeights = new List<double>();
                List<double> dataBiases = new List<double>();
                int weightIndex = 0;
                int biaseIndex = 0;
                dataWeights = netData.NeuralNetworkDna.Weights;
                dataBiases = netData.NeuralNetworkDna.Biases;
                InstanceDNA = netData.NeuralNetworkDna;
                dataWeights = RandomizeNeuralNetworkDNA(dataWeights, trainingRate);
                InstanceDNA.Weights = dataWeights;
                for (int i = 0; i < InputLayer.NeuronsInLayer.Count; i++)
                {
                    InputLayer.NeuronsInLayer[i].Weights.Clear();
                    InputLayer.NeuronsInLayer[i].Weights.Add(dataWeights[weightIndex]);
                    weightIndex++;
                }
                InputLayer.LayerBias = dataBiases[biaseIndex];
                biaseIndex++;
                for (int i = 0; i < HiddenLayers.Count; i++)
                {
                    for (int j = 0; j < HiddenLayers[i].NeuronsInLayer.Count; j++)
                    {
                        HiddenLayers[i].NeuronsInLayer[j].Weights.Clear();
                        for (int k = 0; k < HiddenLayers[i].NeuronsInLayer[j].PreviousNeurons; k++)
                        {
                            HiddenLayers[i].NeuronsInLayer[j].Weights.Add(dataWeights[weightIndex]);
                            weightIndex++;
                        }
                    
                    }
                    HiddenLayers[i].LayerBias = dataBiases[biaseIndex];
                    biaseIndex++;
                }
                for (int j = 0; j < OutputLayer.NeuronsInLayer.Count; j++)
                {
                    OutputLayer.NeuronsInLayer[j].Weights.Clear();
                    for (int k = 0; k < OutputLayer.NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        OutputLayer.NeuronsInLayer[j].Weights.Add(dataWeights[weightIndex]);
                        weightIndex++;
                    }
                }
                OutputLayer.LayerBias = dataBiases[biaseIndex];
                }
            }
            else
            {
                Debug.Log("NetData not found !");
                SetDNARandom(NeuralNetConnections);
            }
            
        }
        private List<double> RandomizeNeuralNetworkDNA(List<double> dataWeights, double TrainingRate)
        {
            List<double> randomized = new List<double>();
            for (int i = 0; i < dataWeights.Count; i++)
            {
                var result = Random.Range((float) dataWeights[i] - (float) TrainingRate, (float) dataWeights[i] + (float) TrainingRate);
                randomized.Add(result);
            }
            
            return randomized;
        }
        private void WaitForAllOutputResults(double result, int OutputIndex)
        {
            if (_countResultsEntry < _outputCount)
            {
                _cycleResults[OutputIndex] = result;
                _countResultsEntry++;
            }
            else
            {
                ValuesToNetOutput(_cycleResults);
                if (NeuralNetworkManager.NetworkMode == NeuralNetworkManager.ENetworkMode.Train)
                {
                    if (this.NeuralNetworkManager.NetworkFunction == NeuralNetworkManager.ENetworkFunction.ComputeData)
                    {
                        EvaluateNeuralNetworkEfficiency(_cycleResults, NeuralNetworkManager.NetworkFunction, NeuralNetworkManager.AbsoluteValues);
                    }
                }
                _countResultsEntry = 0;
               
            }
        }
        private void ValuesToNetOutput(List<double> results)
        {
            for (int i = 0; i < results.Count; i++)
            {
                NeuralNetworkComponent.NetOutput[i].OutputValue = results[i];
            }
        }
        public void ComputeNeuralNetOutput(double OutputResult, int OutputNeuronIndex)
        {
            WaitForAllOutputResults(OutputResult, OutputNeuronIndex);
        }
        public void RestartInstance(NeuralNetworkManager.ENetworkMode eNetworkMode, NetData netData, bool DNAHasUpgrade, bool forceInstanceDNAReset)
        {
            if (eNetworkMode == NeuralNetworkManager.ENetworkMode.Train)
            {
                if (DNAHasUpgrade || forceInstanceDNAReset)
                {
                    SetDNAFromData(netData, NeuralNetworkManager.ForceGeneticsRandomization, NeuralNetworkManager.TrainingRate);
                    Debug.Log("Set DNA from Data");
                }
            }
            // if no DNA upgrade
            NeuralNetworkComponent.Restart();
        }
        private void StartInstance(NeuralNetworkManager.ENetworkMode eNetworkMode)
        {
            EpochsRemaining--;
            if (eNetworkMode == NeuralNetworkManager.ENetworkMode.Train)
            {
                IsTraining = true;
                IsExecuting = false;
            }
            if (eNetworkMode == NeuralNetworkManager.ENetworkMode.Execute)
            {
                IsTraining = false;
                IsExecuting = true;
            }
        }
        public void OnInstanceEnd(List<double> paramatersForEvaluation, NeuralNetworkController controller)
        {
            if (NeuralNetworkManager.NetworkMode == NeuralNetworkManager.ENetworkMode.Train)
            {
                if (NeuralNetworkManager.NetworkFunction ==
                    NeuralNetwork.NeuralNetworkManager.ENetworkFunction.ControlEntity)
                {
                    EvaluateNeuralNetworkEfficiency(paramatersForEvaluation, NeuralNetworkManager.NetworkFunction, NeuralNetworkManager.AbsoluteValues);
                
                }
            }
            else
            {
                Debug.Log("Bypass Training Feedback Evaluation");
                NeuralNetworkManager.BypassTrainingFeedBackEvaluationAndStartNextEpoch();
            }
            
        }
        private void EvaluateNeuralNetworkEfficiency(List<double> externalParameters, NeuralNetworkManager.ENetworkFunction networkFunction, NeuralNetworkManager.EAbsoluteValues absoluteValues)
        {
            List<double> allCoeffs = new List<double>();
            double computedCoeff = 0;
            double divider = 0;
            if (networkFunction == NeuralNetwork.NeuralNetworkManager.ENetworkFunction.ComputeData)
            {
                if (absoluteValues == NeuralNetworkManager.EAbsoluteValues.Yes)
                {
                    for (int i = 0; i < externalParameters.Count; i++)
                    {
                        divider++;// += NeuralNetworkManager.WantedResultPerOutput[i];
                        allCoeffs.Add(Mathf.Abs((float)NeuralNetworkManager.InternalParameters[i] - (float)externalParameters[i]));
                        computedCoeff += allCoeffs[i];
                    }
                    computedCoeff /= divider;
                }
                else
                {
                    for (int i = 0; i < externalParameters.Count; i++)
                    {
                        divider++;// += NeuralNetworkManager.WantedResultPerOutput[i];
                        allCoeffs.Add((float)NeuralNetworkManager.InternalParameters[i] - (float)externalParameters[i]);
                        computedCoeff += allCoeffs[i];
                    }
                    computedCoeff /= divider;
                }
            }
            if (networkFunction == NeuralNetwork.NeuralNetworkManager.ENetworkFunction.ControlEntity)
            {
                if (absoluteValues == NeuralNetworkManager.EAbsoluteValues.Yes)
                {
                    for (int i = 0; i < externalParameters.Count; i++)
                    {
                        divider++;// += NeuralNetworkManager.WantedResultPerOutput[i];
                        allCoeffs.Add(Mathf.Abs((float)externalParameters[i]));
                        computedCoeff += allCoeffs[i];
                    }
                    computedCoeff /= divider;
                }
                else
                {
                    for (int i = 0; i < externalParameters.Count; i++)
                    {
                        divider++;// += NeuralNetworkManager.WantedResultPerOutput[i];
                        allCoeffs.Add((float)externalParameters[i]);
                        computedCoeff += allCoeffs[i];
                    }
                    computedCoeff /= divider;
                } 
            }
            // for exemple, I want the result of the network to be as close as the wanted value as possible
            NeuralNetworkManager.OnInstanceHasEnd(this, computedCoeff, externalParameters, allCoeffs);
        }
        
        [Serializable]
        public struct Layer
        {
            public int NeuronCount;
            public double LayerBias;
        }
        [Serializable]
        public struct DNA
        {
            public List<double> Weights;
            public List<double> Biases;
        }

    }
}