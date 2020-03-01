using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Utils;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NeuralNetwork
{
   
    public class NeuralNet : MonoBehaviour
    {
        [Header("Neural Network Architecture")]
        
        [HideInInspector] 
        public NeuralNetworkManager NeuralNetworkManager;
        
        public NetLayerConstructor inputLayerConstructor = new NetLayerConstructor();
        public List<NetLayerConstructor> HiddenLayers = new List<NetLayerConstructor>();
        public NetLayerConstructor outputLayerConstructor = new NetLayerConstructor();

        [HideInInspector] public NetData _NetData;
        [Header("DNA Display")]
        public DNA InstanceDNA = new DNA();
        
        [Header("Network Execution")]
        public int InstanceID;
        public bool IsExecuting;
        public bool IsTraining;
     
        [Header("Input From World and Output To World")]
        public NeuralNetController Controller;
        public bool inputStreamOn;
        public List<double> ExternalInputs = new List<double>();
        public List<double> OutputToExternal = new List<double>();
        
        #region Initialisation
        public void InitializeNeuralNetwork(NeuralNetworkManager.ERunningMode eRunningMode, int epochs = 0, int instanceID = 0, NeuralNetworkManager neuralNetworkManager = null, NetData netData = null)
        {
            Debug.Log("Starting Initialisation");
            _NetData = netData;
            NeuralNetworkManager = neuralNetworkManager;
            InstanceID = instanceID;
            
            if (OutputToExternal.Count == 0)
            {
                for (int i = 0; i <  outputLayerConstructor.Neurons; i++)
                {
                   OutputToExternal.Add(0);
                }
            }
            if (ExternalInputs.Count == 0)
            {
                for (int i = 0; i <  inputLayerConstructor.Neurons; i++)
                {
                    ExternalInputs.Add(0);
                }
            }
            
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train && !NeuralNetworkManager.NewTraining )
            {
                GetWeights()
            }

            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train &&
                !NeuralNetworkManager.NewTraining)
            {
                InitializeWeights();
            }
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Execute)
            {
               GetWeights()
            }
            SetInstanceRunningMode(eRunningMode);
           
        }

        private int WeightsCount()
        {
            int nbr = 0;
            int hiddensCount = HiddenLayers.Count;
            nbr = (inputLayerConstructor.Neurons * HiddenLayers[0].Neurons); //Input to first
            if (hiddensCount == 1)
            {
                nbr += HiddenLayers[0].Neurons * HiddenLayers[1].Neurons;
                nbr += HiddenLayers[1].Neurons * outputLayerConstructor.Neurons;
            }
            if (hiddensCount > 1)
            {
                for (int i = 0; i < hiddensCount-2; i++)
                {
                    nbr += HiddenLayers[i].Neurons * HiddenLayers[i + 1].Neurons;
                }
                nbr += HiddenLayers[hiddensCount - 2].Neurons * outputLayerConstructor.Neurons;
            }
            nbr += HiddenLayers[hiddensCount - 1].Neurons + outputLayerConstructor.Neurons;
            return nbr;
        }
        private void InitializeWeights(int nbr)
        {
            
            double[] initialWeights = new double[nbr];
            
            
            
            SetWeights(initialWeights);
        }

        private void SetWeights(double[] weights)
        {
           
            
        }

        private double[] GetWeightsAndBiases()
        {
            double[] weights = new double[];

            return weights;
        }
        
        #endregion
        
        #region DNAManaging
        
        public void GetWeightsAndBiasesFromData(NetData netData, NeuralNetworkManager.ELearningLogic learningLogic, double trainingRate)
        {
            if (netData.HasData)
            {
                
                
            }
            else
            {
                InitializeWeights();
            }

        }
        
        private void GetInstanceDNA(NeuralNet instanceNetwork, out double[] _weights, out double[] _biases)
        {
            List<double> biases = new List<double>();
            List<double> weights = new List<double>();
            for (int i = 0; i < instanceNetwork.inputLayerConstructor.NeuronsInLayer.Count; i++)
            {
                weights.Add(instanceNetwork.inputLayerConstructor.NeuronsInLayer[i].Weights[0]);
            }
            biases.Add(instanceNetwork.inputLayerConstructor.LayerBias);
            for (int i = 0; i < instanceNetwork.HiddenLayers.Count; i++)
            {
                for (int j = 0; j < instanceNetwork.HiddenLayers[i].NeuronsInLayer.Count; j++)
                {
                    for (int k = 0; k < instanceNetwork.HiddenLayers[i].NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        weights.Add(instanceNetwork.HiddenLayers[i].NeuronsInLayer[j].Weights[k]);
                    }
                }
                biases.Add(instanceNetwork.HiddenLayers[i].LayerBias);
            }
            
            for (int j = 0; j < instanceNetwork.outputLayerConstructor.NeuronsInLayer.Count; j++)
            {
                for (int k = 0; k < instanceNetwork.outputLayerConstructor.NeuronsInLayer[j].PreviousNeurons; k++)
                {
                    weights.Add(instanceNetwork.outputLayerConstructor.NeuronsInLayer[j].Weights[k]);
                }
            }
            biases.Add(instanceNetwork.outputLayerConstructor.LayerBias);
            
            _weights = weights.ToArray();
            _biases = biases.ToArray();
        } 
        
        #endregion

        private double[] ExecuteSequence(double[] inputs)
        {
            double[] outputs = new double[OutputToExternal.Count];
            double computedOutput = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                
            }
                    
                    
                    
                    
                    
                return outputs;
        }
        
        #region Genetic
        public void Genetic_OnInstanceEnd(List<NeuralNetworkPerformanceSolver> paramatersForEvaluation) // Triggers Only With Genetic Learning
        {
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                Genetic_ComputePerformanceIndex(paramatersForEvaluation);
            }
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                NeuralNetworkManager.ForceStartNextEpoch();
            }
            
        }

        public void Genetic_ComputePerformanceIndex(
            List<NeuralNetworkPerformanceSolver> errorParameters)
        {
            // Compare Values to Actual DNA evaluation parameters values
            double performanceIndex = 0; // so we need to set-up a performance value wich will compare values from ActualBestDna parameters to this instance parameters
            List<NeuralNetworkPerformanceSolver> actualDnaSolvers = new List<NeuralNetworkPerformanceSolver>();
            if (NeuralNetworkManager.ActualBestDNA.PerformanceSolvers.Count == errorParameters.Count)
            {
                actualDnaSolvers =
                    NeuralNetworkManager.ActualBestDNA.PerformanceSolvers; // getting the values from ActualBestDna
            }
            else
            {
                Debug.Log("NeuralNetworkManager Actual DNA has not been initialized. Error.");
            }
            for (int i = 0; i < errorParameters.Count; i++)
            {
                double perfIndOfi = 0;
                double actualDnaParam = actualDnaSolvers[i].EvaluationParameter;
                if (errorParameters[i].ParameterType == EParameterType.ShouldBeInferior
                ) // this instance value should be inferior as best dna value to consider it as better
                {
                        perfIndOfi = actualDnaParam -
                                     errorParameters[i].EvaluationParameter; // value will be positive
                        perfIndOfi *= errorParameters[i].EvaluationParameterWeight; // applying weight coeeficient
                        performanceIndex += perfIndOfi;
                    
                   
                }
                if (errorParameters[i].ParameterType == EParameterType.ShouldBeSuperior) // this instance value should be superior as best dna value to consider it as better
                {
                        perfIndOfi = errorParameters[i].EvaluationParameter -
                                     actualDnaParam;
                        perfIndOfi *= errorParameters[i].EvaluationParameterWeight;
                        performanceIndex += perfIndOfi;
                }
                if (errorParameters[i].ParameterType == EParameterType.ConvergeToExpectedValue)
                {
                    perfIndOfi = Mathf.Abs((float)errorParameters[i].EvaluationParameter - (float)errorParameters[i].ExpectedValue);
                    if (perfIndOfi < Mathf.Abs((float) actualDnaParam -
                                               (float) errorParameters[i].ExpectedValue))
                    {
                        performanceIndex += perfIndOfi;
                    }
                }
            }
            bool instanceHasBestDna = false;
            if (performanceIndex > 0)
            {
                instanceHasBestDna = true;
            }
            else
            {
                Debug.Log("Performance index is inferior than 0.");
            }
            Debug.Log("NeuralNet. Instance Ending");
            NeuralNetworkManager.Genetic_OnInstanceCompare(this, performanceIndex, errorParameters, instanceHasBestDna);

        }
        public void Genetic_NeuralNetRestart(NeuralNetworkManager.ERunningMode eRunningMode, NetData netData, bool DNAHasUpgrade, bool forceInstanceDNAReset)
        {
            if (NeuralNetworkManager.LearningLogic == NeuralNetworkManager.ELearningLogic.Genetic)
            {
                if (eRunningMode == NeuralNetworkManager.ERunningMode.Train && DNAHasUpgrade ||
                    eRunningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train && forceInstanceDNAReset)
                {

                    GetWeightsAndBiasesFromData(netData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);
                    Debug.Log("Set DNA from Data");
                }
                if(!gameObject.activeSelf) gameObject.SetActive(true);
                // if no DNA upgrade
                Controller.InstanceReset();
            }
        }
        
        private List<double> RandomizeWeightsAndBiasesFromData(List<double> dataWeights, double TrainingRate)
        {
            List<double> randomized = new List<double>();
            for (int i = 0; i < dataWeights.Count; i++)
            {
                var result = Random.Range((float) dataWeights[i] - (float) TrainingRate, (float) dataWeights[i] + (float) TrainingRate);
                randomized.Add(result);
            }
            
            return randomized;
        }
        #endregion
        
        #region BackPropagation
        
        private void BackPropagation_OnSequenceEnd(double[] result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                OutputToExternal[i] = result[i];
            }
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                // BackPropagation Loop
            }
        }
        
        #endregion

        #region Others
        
        private void SetInstanceRunningMode(NeuralNetworkManager.ERunningMode eRunningMode)
        {
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                IsTraining = true;
                IsExecuting = false;
            }
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                IsTraining = false;
                IsExecuting = true;
            }
        }
        
        #endregion
      

    }
}